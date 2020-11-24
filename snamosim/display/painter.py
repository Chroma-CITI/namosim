import logging

from future.utils import with_metaclass
from snamosim.utils.singleton import Singleton


class Painter(with_metaclass(Singleton)):
    def __init__(self, top_level_namespaces=('simulation', 'agent'),
                 display_active=True, rp_active=True, mplt_active=False):
        self._top_level_namespaces = top_level_namespaces
        self._display_active = display_active
        self._rp_active = rp_active
        self._rp_initialized = False
        self._mplt_active = mplt_active
        self._mplt_initialized = False

        if self._display_active:
            if self._rp_active:
                try:
                    # noinspection PyUnresolvedReferences
                    from ros_publisher import RosPublisher

                    self._rp = RosPublisher(self._top_level_namespaces)
                    self._rp_initialized = True
                except ImportError as e:
                    logging.warning(
                        'Could not import ros_publisher module, because importing the {} module failed.'
                        'Visualizations will only be provided through matplotlib and not through RViz.'
                        'Problem sources can be:\n'
                        '- ROS is not installed on your system\n'
                        '- ROS is installed but not in PYTHONPATH '
                        '(most likely because ROS setup.bash file was not sourced, '
                        'typically if you did not launch your code editor from the terminal.)'.format(e.name))
            if self._mplt_active:
                self._mplt = None
                self._mplt_initialized = False

    def _is_rp_available(self):
        if self._display_active and self._rp_active and self._rp_initialized:
            return True
        elif not ():
            return False
        elif self._display_active and self._rp_active and not self._rp_initialized:
            raise RuntimeError('Tried to used RosPublisher singleton but not initialized !')

    def _is_mplt_available(self):
        if self._display_active and self._mplt_active and self._mplt_initialized:
            return True
        elif not ():
            return False
        elif self._display_active and self._mplt_active and not self._mplt_initialized:
            raise RuntimeError('Tried to used MatplotlibPainter singleton but not initialized !')

    def clean_all(self):
        if self._is_rp_available():
            self._rp.clean_all()
        if self._is_mplt_available():
            self._mplt.clean_all()
