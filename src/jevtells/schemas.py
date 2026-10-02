from pydantic import BaseModel
from typing import Any
class Window(BaseModel):
 id:str; index:int; t0:float; t1:float; subtitle:str; prev_subtitle:str=''; shot:str='target'; kind:str='speech'
class State(BaseModel):
 scene:str; speaker:str; subtitle:dict[str,str]; voice:str; measured_actions:list[str]
