from simulator import Simulator


if __name__ == '__main__':

    sim = Simulator(simulation_file_path="../data/simulations/first_level/01_two_rooms_corridor/navigation_only_behavior.yaml")

    sim.run()

    # banner = """
    # Welcome in the S-NAMO simulator !
    #
    # You are within the program now: you can access the simulator's
    # functions simply by calling it's python functions, like so:
    #
    # sim.help()
    #
    # To Exit, press Ctrl+z.
    # """
    #
    # print(banner)
    #
    # embed(globals(), locals())
