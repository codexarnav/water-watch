import os
import time
import json
import logging
import asyncio
from typing import Dict, Optional, Any

from playwright.async_api import async_playwright, Page, BrowserContext
from playwright_stealth import Stealth

from utils import (
    BASE_DIR, 
    IMAGES_DIR, 
    VIDEOS_DIR, 
    AUDIO_DIR,
    get_timestamp_iso, 
    generate_filename,
    save_metadata
)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WhatsAppScraper")

# Constants
USER_DATA_DIR = BASE_DIR / "user_data"
WHATSAPP_WEB_URL = "https://web.whatsapp.com"

class WhatsAppScraper:
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.browser_context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None

    async def start(self):
        """Starts the Playwright browser with persistent context."""
        logger.info("Starting WhatsApp Scraper...")
        self.playwright = await async_playwright().start()
        
        # Use persistent context to save login session
        self.browser_context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR,
            headless=self.headless,
            channel="chrome", # Use installed Chrome if available, else standard chromium
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1280, "height": 720}
        )
        
        self.page = self.browser_context.pages[0]
        
        # Apply stealth
        await Stealth().apply_stealth_async(self.page)
        
        logger.info(f"Navigating to {WHATSAPP_WEB_URL}...")
        await self.page.goto(WHATSAPP_WEB_URL)
        
        # Wait for login (check for a specific element that exists only when logged in)
        # The chat list pane is a good indicator
        try:
            logger.info("Waiting for login... (Please scan QR code if needed)")
            await self.page.wait_for_selector('div[id="pane-side"]', timeout=60000 * 5) # Wait up to 5 mins for login
            logger.info("Login detected!")
        except Exception:
            logger.error("Login timed out. Please restart and scan QR code.")
            await self.stop()
            return

    async def stop(self):
        """Stops the browser."""
        if self.browser_context:
            await self.browser_context.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Scraper stopped.")

    async def monitor(self):
        """Main loop to monitor for new messages."""
        if not self.page:
            return

        logger.info("Monitoring for new messages... (Press Ctrl+C to stop)")
        
        last_check_time = 0
        self.ignored_chats = set()
        processed_count = 0
        MESSAGE_LIMIT = 5
        
        while True:
            if processed_count >= MESSAGE_LIMIT:
                logger.info(f"Reached limit of {MESSAGE_LIMIT} messages. Stopping scraper.")
                break

            try:
                # Heartbeat log
                if time.time() - last_check_time > 10:
                    logger.info(f"Scanning... (Processed: {processed_count}/{MESSAGE_LIMIT}, Ignored old: {len(self.ignored_chats)})")
                    last_check_time = time.time()


                
                unread_badges = await self.page.locator('span[aria-label*="unread"], span[aria-label*="Unread"]').all()
                
                if not unread_badges:
                    await self.page.wait_for_timeout(1000)
                    continue

                # Debug: Print the first one to see what we are looking at
                # try:
                #     first_badge = unread_badges[0]
                #     first_row = first_badge.locator('xpath=./ancestor::div[@role="row"]')
                #     if await first_row.count() > 0:
                #         debug_text = await first_row.inner_text()
                #         debug_lines = debug_text.split('\n')
                #         debug_id = debug_lines[0] if debug_lines else "unknown"
                #         # logger.info(f"Top unread chat is: {debug_id}")
                # except:
                #     pass

                if len(unread_badges) != getattr(self, "last_badge_count", 0):
                    logger.info(f"Found {len(unread_badges)} unread badges")
                    self.last_badge_count = len(unread_badges)
                
                #
                processed_any = False
                
                for badge in unread_badges:
                    
                    row = badge.locator('xpath=./ancestor::div[@role="row"]')
                    
                    if await row.count() == 0:
                        continue
                        
                    row_text = await row.inner_text()
                    lines = row_text.split('\n')
                    
                   
                    
                    # Identifying key (Name + partial text) to track ignored chats
                    chat_id = lines[0] if lines else "unknown"
                    
                    if chat_id in self.ignored_chats:
                        continue
                        
                    
                    is_old = False
                    if "Yesterday" in row_text:
                         is_old = True
                    else:
                        import re
                        if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', row_text):
                            is_old = True
            
                    if is_old:
                        logger.info(f"Ignoring old unread chat: {chat_id}")
                        self.ignored_chats.add(chat_id)
                        continue
                      
            
                    logger.info(f"Processing new message from: {chat_id}")
                    
                    await badge.click()
                    
                    # Wait/Process
                    await self.page.wait_for_timeout(2000)
                    await self.process_current_chat()
                    
                    processed_count += 1
                    logger.info(f"Message {processed_count}/{MESSAGE_LIMIT} processed.")
                    
                    await self.page.wait_for_timeout(2000)
                    
                    # Break to refresh the list (badges might have moved/changed)
                    break
                
                if not processed_any:
                    await self.page.wait_for_timeout(1000)
                    
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await self.page.wait_for_timeout(5000)

    async def process_current_chat(self):
        """Process messages in the currently open chat."""
        
        messages = self.page.locator('div[role="row"]')
        count = await messages.count()
        
        if count == 0:
            logger.warning("No messages found in chat!")
            return
            
        last_message = messages.nth(count - 1)
        
        # Debug
        try:
            msg_html = await last_message.inner_html()
            # logger.info(f"Last message HTML length: {len(msg_html)}")
        except:
            pass
        
       
        # 1. Text
        # Try multiple text selectors as class names vary
        # Update: WhatsApp Web often uses specific classes for text bubbles.
        # We will try a few known ones, and then a fallback to the message container's text.
        text_selectors = [
            'span.selectable-text', 
            'div.copyable-text', 
            'span[dir="ltr"]', 
            '.message-text',
            'div[class*="copyable-text"]'
        ]
        text_content = ""
        
        for sel in text_selectors:
            # We search inside the last message
            text_element = last_message.locator(sel).first
            if await text_element.count() > 0:
                text_content = await text_element.inner_text()
                if text_content:
                    break
        
        # Fallback: If no specific text class found, get all text from the message bubble
        # and try to strip out time/meta info if possible (heuristic)
        if not text_content:
             # Try to find the inner container of the message
             # usually a div with class 'copyable-text' or similar is best, but if that failed...
             # Let's just get the full text of the row and log it for debug
             full_row_text = await last_message.inner_text()
             # logger.warning(f"Fallback text extraction. Row text: {full_row_text[:50]}...")
             
             # Use the full text as a last resort if it's not too long/noisy
             # But usually row text includes time, which is fine to keep for now
             if full_row_text:
                 text_content = full_row_text
            
        
        images = last_message.locator('img[src^="blob:"]')
        videos = last_message.locator('video')
        audio = last_message.locator('audio') # Not common in web DOM, usually generic player
        
        logger.info(f"Content check: Text={bool(text_content)}, Images={await images.count()}, Videos={await videos.count()}")
        
        modality = "text"
        payload = {}
        
        if await videos.count() > 0:
            modality = "video"
            filename = await self.download_attachment(last_message, "video")
            if filename:
                payload["video_uri"] = str(filename)
                
        elif await images.count() > 0:
             
             modality = "image"
             filename = await self.download_attachment(last_message, "image")
             if filename:
                payload["image_uri"] = str(filename)
        
        elif text_content:
            modality = "text"
            payload["text"] = text_content
            
        else:
           
            if await last_message.locator('span[data-icon="audio-play"]').count() > 0:
                modality = "audio"
                filename = await self.download_attachment(last_message, "audio")
                if filename:
                    payload["audio_uri"] = str(filename)

        if payload:
            data = {
                "modality": modality,
                "payload": payload,
                "context": {
                    "timestamp": get_timestamp_iso(),
                    "source": "whatsapp",
                    "geohash": "unknown" 
                }
            }
            save_metadata(data)
            logger.info(f"Processed message: {modality}")

    async def download_attachment(self, message_locator, media_type):
        """Downloads media from a message."""
        try:
            
            if media_type in ["image", "video"]:
                onclick_element = message_locator.locator('div[role="button"]').first
                if await onclick_element.count() == 0:
                     # fallback to the image/video element
                     onclick_element = message_locator.locator('img' if media_type == 'image' else 'video').first
                
                if await onclick_element.count() > 0:
                    await onclick_element.click()
                    
                    # Wait for viewer to open
                    await self.page.wait_for_timeout(500)
                    
                    
                    download_btn = self.page.locator('span[data-icon="download"]')
                    
                    if await download_btn.count() > 0:
                        async with self.page.expect_download() as download_info:
                            await download_btn.click()
                        
                        download = await download_info.value
                        
                        # Save
                        ext = ".mp4" if media_type == "video" else ".jpg" 
                        filename = generate_filename("wa_msg", ext)
                        
                        target_dir = VIDEOS_DIR if media_type == "video" else IMAGES_DIR
                        save_path = target_dir / filename
                        
                        await download.save_as(save_path)
                        
                        # Close viewer - Escape key
                        await self.page.keyboard.press("Escape")
                        
                        return str(save_path.relative_to(BASE_DIR))
            
            return None
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            # Try to close viewer if open
            await self.page.keyboard.press("Escape")
            return None

if __name__ == "__main__":
    scraper = WhatsAppScraper(headless=False)
    
    async def main():
        try:
            await scraper.start()
            # Only monitor if start was successful (page is set)
            if scraper.page:
                await scraper.monitor()
        except asyncio.CancelledError:
            logger.info("Task cancelled")
        finally:
            await scraper.stop()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal error: {e}")
