"""
Pre-initialization for FloorPlan3 task.
FloorPlan3's faucet and light are turned on
"""

class SceneInitializer:
    def __init__(self) -> None:
        pass
        
    def preinit(self, event, controller):
        """Pre-initialize the environment for the task.
    
        Args:
            event: env.event object
            controller: ai2thor.controller object
    
        Returns:
            event: env.event object
        """
    
            # initialization function - autogenerated
    
        event=controller.step(
        action='ToggleObjectOn',
        objectId='Faucet|-02.13|+01.32|-00.81',
        forceAction=True
        )
                        
        event=controller.step(
        action='ToggleObjectOn',
        objectId='LightSwitch|+01.39|+01.57|-00.66',
        forceAction=True
        )
                        
        return event
            