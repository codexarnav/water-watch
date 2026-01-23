"""
Risk forecasting service using retrieval-based prediction
"""
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.qdrant_service import qdrant_service
from services.embedding_service import embedding_service
from config import settings
from qdrant_client import models

logger = logging.getLogger(__name__)


class RiskForecastingService:
    def __init__(self):
        self.high_threshold = settings.RISK_HIGH_THRESHOLD
        self.medium_threshold = settings.RISK_MEDIUM_THRESHOLD
    
    def calculate_risk_score(self, sensor_data: Dict[str, Any]) -> float:
        """
        Calculate risk score based on sensor readings
        Uses thresholds for pH, DO, salinity
        """
        risk_factors = []
        
        # pH risk (optimal: 6.5-8.5)
        ph = sensor_data.get('ph')
        if ph is not None:
            if ph < 6.0 or ph > 9.0:
                risk_factors.append(0.9)
            elif ph < 6.5 or ph > 8.5:
                risk_factors.append(0.6)
            else:
                risk_factors.append(0.2)
        
        # Dissolved Oxygen risk (optimal: > 5 mg/L)
        do = sensor_data.get('dissolved_oxygen')
        if do is not None:
            if do < 2.0:
                risk_factors.append(0.95)
            elif do < 5.0:
                risk_factors.append(0.7)
            else:
                risk_factors.append(0.2)
        
        # Salinity risk (depends on site, but extreme values are bad)
        salinity = sensor_data.get('salinity')
        if salinity is not None:
            if salinity > 10.0:
                risk_factors.append(0.8)
            elif salinity > 5.0:
                risk_factors.append(0.5)
            else:
                risk_factors.append(0.2)
        
        # Water temperature risk (extreme temps)
        water_temp = sensor_data.get('water_temp')
        if water_temp is not None:
            if water_temp < 0 or water_temp > 35:
                risk_factors.append(0.7)
            elif water_temp < 5 or water_temp > 30:
                risk_factors.append(0.4)
            else:
                risk_factors.append(0.1)
        
        # Average risk
        if risk_factors:
            return np.mean(risk_factors)
        return 0.5  # Default moderate risk
    
    def get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to level"""
        if risk_score >= self.high_threshold:
            return "high"
        elif risk_score >= self.medium_threshold:
            return "medium"
        return "low"
    
    def analyze_trends(
        self,
        site_id: str,
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Analyze trends for pH, DO, and salinity"""
        trends = {
            "ph_trend": "stable",
            "do_trend": "stable",
            "salinity_trend": "stable"
        }
        
        if len(historical_data) < 2:
            return trends
        
        # Sort by timestamp
        sorted_data = sorted(
            historical_data,
            key=lambda x: x.get('timestamp', datetime.min)
        )
        
        # Analyze pH trend
        ph_values = [d.get('ph') for d in sorted_data if d.get('ph') is not None]
        if len(ph_values) >= 2:
            if ph_values[-1] < ph_values[0] - 0.5:
                trends['ph_trend'] = "decreasing"
            elif ph_values[-1] > ph_values[0] + 0.5:
                trends['ph_trend'] = "increasing"
        
        # Analyze DO trend
        do_values = [d.get('dissolved_oxygen') for d in sorted_data if d.get('dissolved_oxygen') is not None]
        if len(do_values) >= 2:
            if do_values[-1] < do_values[0] - 1.0:
                trends['do_trend'] = "decreasing"
            elif do_values[-1] > do_values[0] + 1.0:
                trends['do_trend'] = "increasing"
        
        # Analyze salinity trend
        salinity_values = [d.get('salinity') for d in sorted_data if d.get('salinity') is not None]
        if len(salinity_values) >= 2:
            if salinity_values[-1] < salinity_values[0] - 0.5:
                trends['salinity_trend'] = "decreasing"
            elif salinity_values[-1] > salinity_values[0] + 0.5:
                trends['salinity_trend'] = "increasing"
        
        return trends
    
    def generate_recommendations(
        self,
        sensor_data: Dict[str, Any],
        risk_score: float,
        trends: Dict[str, str]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # pH recommendations
        ph = sensor_data.get('ph')
        if ph is not None:
            if ph < 6.5:
                recommendations.append("⚠️ pH is low - consider adding alkaline substances")
            elif ph > 8.5:
                recommendations.append("⚠️ pH is high - investigate potential contamination")
            
            if trends['ph_trend'] == "decreasing":
                recommendations.append("📉 pH is decreasing - monitor closely")
            elif trends['ph_trend'] == "increasing":
                recommendations.append("📈 pH is increasing - check for algal blooms")
        
        # DO recommendations
        do = sensor_data.get('dissolved_oxygen')
        if do is not None:
            if do < 5.0:
                recommendations.append("💧 Low dissolved oxygen - increase aeration")
            
            if trends['do_trend'] == "decreasing":
                recommendations.append("📉 Dissolved oxygen decreasing - urgent action needed")
        
        # Salinity recommendations
        salinity = sensor_data.get('salinity')
        if salinity is not None:
            if salinity > 5.0:
                recommendations.append("🧂 High salinity detected - check for saltwater intrusion")
        
        # General recommendations based on risk
        if risk_score >= self.high_threshold:
            recommendations.append("🚨 HIGH RISK - Immediate investigation required")
            recommendations.append("📞 Contact water quality experts")
        elif risk_score >= self.medium_threshold:
            recommendations.append("⚠️ MEDIUM RISK - Increase monitoring frequency")
        
        if not recommendations:
            recommendations.append("✅ Water quality within acceptable range")
        
        return recommendations
    
    async def forecast_risk(
        self,
        site_id: str,
        current_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Forecast risk using recent time-series data from Qdrant for specific site.
        """
        try:
            # Calculate current risk
            risk_score = self.calculate_risk_score(current_data)
            risk_level = self.get_risk_level(risk_score)
            
            # Retrieve RECENT history for this specific site (Time Series)
            # We use scroll to get latest points sorted by timestamp desc
            qdrant_service._initialize()
            if not qdrant_service.client:
                raise Exception("Qdrant client not available")

            recent_points, _ = qdrant_service.client.scroll(
                collection_name=qdrant_service.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="site_id",
                            match=models.MatchValue(value=site_id)
                        ),
                        models.FieldCondition(
                             key="type",
                             match=models.MatchValue(value="sensor")
                        )
                    ]
                ),
                limit=20,
                with_payload=True,
                with_vectors=False,
                order_by=models.OrderBy(
                    key="timestamp",
                    direction="desc"
                )
            )
            
            # Extract historical payloads
            historical_data = [point.payload for point in recent_points]
            
            # Analyze trends
            trends_analysis = self.analyze_trends(site_id, historical_data)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(
                current_data,
                risk_score,
                trends_analysis
            )
            
            return {
                "site_id": site_id,
                "risk_level": risk_level,
                "risk_score": round(risk_score, 3),
                "predictions": trends_analysis, 
                "recommendations": recommendations,
                "timestamp": datetime.utcnow()
            }
        except Exception as e:
            logger.error(f"Error forecasting risk for {site_id}: {e}")
            return {
                "site_id": site_id,
                "risk_level": "unknown",
                "risk_score": 0.5,
                "predictions": {
                    "ph_trend": "unknown",
                    "do_trend": "unknown",
                    "salinity_trend": "unknown",
                    "ph_history": [],
                    "do_history": [],
                    "salinity_history": []
                },
                "recommendations": ["Error calculating risk - please check manually"],
                "timestamp": datetime.utcnow()
            }

    def analyze_trends(
        self,
        site_id: str,
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze trends for pH, DO, and salinity and return history for graphing"""
        trends = {
            "ph_trend": "stable",
            "do_trend": "stable",
            "salinity_trend": "stable",
            "ph_history": [],
            "do_history": [],
            "salinity_history": []
        }
        
        if not historical_data:
            return trends
            
        # Sort by timestamp ascending for graph
        sorted_data = sorted(
            historical_data,
            key=lambda x: x.get('timestamp', datetime.min.isoformat())
        )
        
        # Helper to extract points
        def extract_points(key):
            points = []
            values = []
            for d in sorted_data:
                if d.get(key) is not None:
                    try:
                        ts = d.get('timestamp')
                        # Ensure ISO format parsing
                        if isinstance(ts, str):
                            ts_obj = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        else:
                            ts_obj = ts
                        points.append({"timestamp": ts_obj, "value": float(d[key])})
                        values.append(float(d[key]))
                    except (ValueError, TypeError):
                        continue
            return points, values

        ph_points, ph_values = extract_points('ph')
        do_points, do_values = extract_points('dissolved_oxygen')
        sal_points, sal_values = extract_points('salinity')
        
        trends['ph_history'] = ph_points
        trends['do_history'] = do_points
        trends['salinity_history'] = sal_points

        # Calculate simple trends (last vs avg of previous 3)
        if len(ph_values) >= 2:
            if ph_values[-1] < ph_values[0] - 0.5:
                trends['ph_trend'] = "decreasing"
            elif ph_values[-1] > ph_values[0] + 0.5:
                trends['ph_trend'] = "increasing"

        if len(do_values) >= 2:
            if do_values[-1] < do_values[0] - 1.0:
                trends['do_trend'] = "decreasing"
            elif do_values[-1] > do_values[0] + 1.0:
                trends['do_trend'] = "increasing"

        if len(sal_values) >= 2:
            if sal_values[-1] < sal_values[0] - 0.5:
                trends['salinity_trend'] = "decreasing"
            elif sal_values[-1] > sal_values[0] + 0.5:
                trends['salinity_trend'] = "increasing"
        
        return trends


# Global instance
risk_forecasting_service = RiskForecastingService()
