from rl_studio.agents.tasks_type import TasksType
from rl_studio.agents.frameworks_type import FrameworksType
from rl_studio.algorithms.algorithms_type import AlgorithmsType
from rl_studio.envs.gazebo.f1.exceptions import NoValidEnvironmentType


class Carla:
    def __new__(cls, **environment):

        algorithm = environment["algorithm"]
        task = environment["task"]
        framework = environment["framework"]
        weather = environment["weather"]
        traffic_pedestrians = environment["traffic_pedestrians"]

        if framework == "baselines":
        # if framework == FrameworksType.BASELINES:
            from rl_studio.envs.carla.followlane.followlane_carla_sb import (
                FollowLaneStaticWeatherNoTraffic,
            )

            return FollowLaneStaticWeatherNoTraffic(**environment)

        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and algorithm == AlgorithmsType.QLEARN.value
            and weather != "dynamic"
            and traffic_pedestrians is False
        ):
            from rl_studio.envs.carla.followlane.followlane_qlearn import (
                FollowLaneQlearnStaticWeatherNoTraffic,
            )

            return FollowLaneQlearnStaticWeatherNoTraffic(**environment)

        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and algorithm == AlgorithmsType.PPO_CONTINIUOUS.value
            and weather != "dynamic"
            and traffic_pedestrians is False
        ):
            from rl_studio.envs.carla.followlane.followlane_carla import (
                FollowLaneStaticWeatherNoTraffic,
            )

            return FollowLaneStaticWeatherNoTraffic(**environment)

        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and algorithm == AlgorithmsType.DDPG.value
            and weather != "dynamic"
            and traffic_pedestrians is False
        ):
            from rl_studio.envs.carla.followlane.followlane_carla import (
                FollowLaneStaticWeatherNoTraffic,
            )

            return FollowLaneStaticWeatherNoTraffic(**environment)

        elif (
            task == TasksType.FOLLOWLANECARLA.value
            and algorithm in [AlgorithmsType.MANUAL.value, AlgorithmsType.AUTO.value]
            and weather != "dynamic"
            and traffic_pedestrians is False
        ):
            from rl_studio.envs.carla.followlane.followlane_carla import (
                FollowLaneStaticWeatherNoTraffic,
            )

            return FollowLaneStaticWeatherNoTraffic(**environment)

        else:
            raise NoValidEnvironmentType(task)