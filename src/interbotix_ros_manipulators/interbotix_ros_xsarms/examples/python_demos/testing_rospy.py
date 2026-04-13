from interbotix_xs_modules.arm import InterbotixManipulatorXS

class InterbotixTest:

    def __init__(self): 

        print("\nInitializing robot...")
        self.bot = InterbotixManipulatorXS(
            robot_model=robot_model,
            robot_name=robot_name,
            moving_time=2.0,
            accel_time=0.5,
            init_node=False
        )

        self.gripper_info = self.bot.gripper.gripper_state()

def main():
    test = InterbotixTest()


if __name__ == '__main__':
    main()