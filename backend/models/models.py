from sqlalchemy import Column, Integer, String, Float, Text, JSON, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from db.database_connection import Base
from datetime import datetime

class User(Base):
    __tablename__='users'

    id=Column(Integer,primary_key=True, index=True)
    username=Column(String, unique=True, index=True)
    email=Column(String, unique=True, index=True)
    password=Column(String)
    location=Column(String)

class Userinput(Base):
    __tablename__='userinput'

    id=Column(Integer,primary_key=True,index=True)
    user_id=Column(Integer,ForeignKey('users.id'))
    input_data_type=Column(String)
    input_data=Column(Text)
    timestamp=Column(DateTime, default=datetime.utcnow)

class Recommendtion(Base):
    __tablename__='recommendations'

    id=Column(Integer,primary_key=True,index=True)
    user_id=Column(Integer,ForeignKey('users.id'))
    recommendation_data=Column(JSON)
    timestamp=Column(DateTime, default=datetime.utcnow)

class Forecasting(Base):
    __tablename__='forecasting'

    id=Column(Integer,primary_key=True,index=True)
    user_id=Column(Integer,ForeignKey('users.id'))
    forecasting_data=Column(JSON)
    timestamp=Column(DateTime, default=datetime.utcnow)


class Government_Offices(Base):
    __tablename__='government_offices'

    id=Column(Integer,primary_key=True,index=True)
    office_name=Column(String)
    location=Column(String)
    contact_info=Column(String)

class Progress_Tracking(Base):
    __tablename__='progress_tracking'

    id=Column(Integer,primary_key=True,index=True)
    user_id=Column(Integer,ForeignKey('users.id'))
    progress_data=Column(JSON)
    timestamp=Column(DateTime, default=datetime.utcnow)



