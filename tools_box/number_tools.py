from pydantic import BaseModel, Field
from typing import Dict
import random
class RandomParams(BaseModel):
    min_val:int=Field(1,description="Minimum value")
    max_val:int=Field(1000,description="Maximum value")
    count:int=Field(1,description="How many numbers to generate")

def generate_random_tool(params:RandomParams,context:Dict):
    min_val=params.min_val
    max_val=params.max_val
    count=params.count

    random_list=[]
    if min_val>max_val:
        return {
                "Error":"Maximum value is less the minimum value"
        }
    else:
        for i in range(count):
            random_num=random.randint(min_val,max_val)
            random_list.append(random_num)
    
        return {
            "numbers":random_list,
            "count":count
        }

def register_number_tools(registry):
    "Register random generator tools that works with number with given registry"
    registry.register_tools(
        name:"random_generator",
        param_model:RandomParams,
        function:generate_random_tool,
        description:"generate random of numbers"
    )
        


    

