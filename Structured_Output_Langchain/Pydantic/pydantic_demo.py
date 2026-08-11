from pydantic import BaseModel, EmailStr # emailstr is a built in valdation that of course checks if the email is valid or not, you can also create your own custom validation using pydantic
from typing import Optional
from pydantic import Feild

class Student(BaseModel):
    name: str = 'gurpreet'    # default value
    age: Optional[int] = 18  # you need to set a default value in the case of a optional feild
    cgpa: float = Feild(gt = 0, lt = 10, default  = 6, description = "A decimal value representing the cgpa of a student") # gt means greater than and lt means less than, so this feild will only accept values between 0 and 10

new_student = {'name':32}
student = Student(**new_student)

student

# here we are able to do data validation 


# pydantic can also do implicit type coercing, for example if you pass a string to an integer field, pydantic will try to convert it to an integer. If it fails, it will raise a validation error.