from typing import Any, List, Optional, Union
import datetime

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class DataInputSchema(BaseModel):
    age: Optional[int]
    dailyrate: Optional[int]
    distancefromhome: Optional[int]
    education: Optional[int]
    employeecount: Optional[int]
    employeenumber: Optional[int]
    environmentsatisfaction: Optional[int]
    hourlyrate: Optional[int]
    jobinvolvement: Optional[int]
    joblevel: Optional[int]
    jobsatisfaction: Optional[int]
    monthlyincome: Optional[int]
    monthlyrate: Optional[int]
    numcompaniesworked: Optional[int]
    percentsalaryhike: Optional[int]
    performancerating: Optional[int]
    relationshipsatisfaction: Optional[int]
    standardhours: Optional[int]
    stockoptionlevel: Optional[int]
    totalworkingyears: Optional[int]
    trainingtimeslastyear: Optional[int]
    worklifebalance: Optional[int]
    yearsatcompany: Optional[int]
    yearsincurrentrole: Optional[int]
    yearssincelastpromotion: Optional[int]
    yearswithcurrmanager: Optional[int]

    businesstravel: Optional[str]
    department: Optional[str]
    educationfield: Optional[str]
    gender: Optional[str]
    jobrole: Optional[str]
    maritalstatus: Optional[str]
    overtime: Optional[str]
    over18: Optional[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
