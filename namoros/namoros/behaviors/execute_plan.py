import copy
import typing as t

import py_trees
from kobuki_ros_interfaces.msg import Sound
from namoros_msgs.msg._namo_path import NamoPath
from py_trees.behaviour import Behaviour
from py_trees.composites import Selector, Sequence
from py_trees.trees import BehaviourTree

from namoros.behavior_node import NamoBehaviorNode
from namoros.behaviors.approach import Approach
from namoros.behaviors.clear_global_costmap import ClearGlobalCostmap
from namoros.behaviors.clear_local_costmap import ClearLocalCostmap
from namoros.behaviors.detect_conflicts_guard import DetectConflictsGuard
from namoros.behaviors.face_obstacle import FaceObstacle
from namoros.behaviors.follow_path import FollowPath
from namoros.behaviors.pause import Pause
from namoros.behaviors.play_sound import PlaySound
from namoros.behaviors.redetect_obstacle import RedetectObstacle
from namoros.behaviors.release import Release
from namoros.behaviors.TriggerReplan import TriggerReplan


class ExecuteNamoPlan(py_trees.behaviour.Behaviour):
    def __init__(self, node: NamoBehaviorNode):
        super().__init__(name="execute_namo_plan")
        self.node = node
        self.tree: BehaviourTree | None = None

    def create_conflict_handling_subtree(self, node: NamoBehaviorNode) -> Behaviour:
        handle_conflicts_root = Selector(
            name="handle_conflicts_root",
            memory=True,
            children=[
                DetectConflictsGuard(node=node),
                TriggerReplan(node=node, update_plan=True),
            ],
        )

        return handle_conflicts_root

    def create_postpone_tree(self, seconds: float) -> Behaviour:
        postpone_root = Sequence(
            name="postpone_root",
            memory=True,
            children=[
                Pause(seconds=seconds, node=self.node),
                TriggerReplan(node=self.node, update_plan=True),
            ],
        )
        return postpone_root

    def create_grab_tree(self, path: NamoPath):
        if not self.node.plan:
            raise Exception("No plan")
        if not path.is_transfer:
            raise Exception("Not a transfer path")
        obstacle_id: str = path.obstacle_id  # type: ignore
        root = py_trees.composites.Sequence(
            name="grab_seq",
            memory=True,
            children=[
                PlaySound(node=self.node, sound=Sound.CLEANINGSTART),
                ClearGlobalCostmap(node=self.node),
                ClearLocalCostmap(node=self.node),
                FaceObstacle(node=self.node, obstacle_id=obstacle_id),
                # Grab(node=self.node, path=path),
                Approach(node=self.node, obstacle_id=obstacle_id),
            ],
        )
        return root

    def create_release_tree(self, path: NamoPath):
        if not self.node.plan:
            raise Exception("No plan")
        obstacle_id: str = path.obstacle_id  # type: ignore

        root = py_trees.composites.Sequence(
            name="release",
            memory=True,
            children=[
                PlaySound(node=self.node, sound=Sound.CLEANINGSTART),
                Release(node=self.node, path=path),
                # Pause(seconds=3),
                RedetectObstacle(node=self.node, obstacle_id=obstacle_id),
            ],
        )
        return root

    def create_sub_tree(self):
        if not self.node.plan:
            raise Exception("No plan")

        root = py_trees.composites.Sequence(name="execute_plan", memory=True)

        if self.node.plan.postpone_steps > 0:
            root.add_child(
                self.create_postpone_tree(seconds=self.node.plan.postpone_steps * 0.5)
            )
            return BehaviourTree(root)

        paths: t.List[NamoPath] = self.node.plan.paths  # type: ignore
        for namo_path in paths:
            nav_path = copy.deepcopy(namo_path.path)
            if namo_path.is_transfer:
                assert namo_path.obstacle_id
                grab = self.create_grab_tree(namo_path)
                root.add_child(grab)
                # remove the pre-grab and post-release poses since those are executed manually
                nav_path.poses = nav_path.poses[1:-1]  # type: ignore
            follow_path = FollowPath(
                node=self.node,
                namo_path=namo_path,
                synchronize_planner=True,
                is_evasion=namo_path.is_evasion,
            )

            follow_path_seq = Sequence(
                "follow_path_seq",
                memory=False,
                children=[
                    self.create_conflict_handling_subtree(self.node),
                    follow_path,
                ],
            )

            root.add_children([follow_path_seq])

            if namo_path.is_evasion:
                root.add_child(TriggerReplan(node=self.node))
            if namo_path.is_transfer:
                release = self.create_release_tree(path=namo_path)
                root.add_child(release)
        return BehaviourTree(root)

    def initialise(self):
        if not self.node.plan:
            raise Exception("No plan")
        self.tree = self.create_sub_tree()
        # py_trees.display.render_dot_tree(
        #     self.tree.root, name="execute_plan_tree", target_directory="."
        # )

    def update(self):
        if not self.tree:
            raise Exception("No tree")
        self.tree.tick()
        self.status = self.tree.root.status
        return self.status
