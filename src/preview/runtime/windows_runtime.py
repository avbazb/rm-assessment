"""Keep this windowless real-time process out of Windows automatic EcoQoS."""
import os


def configure_cpu_scheduling():
    if os.name != 'nt':
        return {'high_qos': False, 'platform': os.name}
    import ctypes
    from ctypes import wintypes

    class PowerThrottling(ctypes.Structure):
        _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG),
                    ('StateMask', wintypes.ULONG)]

    try:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                 ctypes.c_void_p, wintypes.DWORD]
        kernel.SetProcessInformation.restype = wintypes.BOOL
        # ProcessPowerThrottling=4, EXECUTION_SPEED=1. Control=0 leaves
        # scheduling system-managed; Control=1 / State=0 explicitly opts out.
        state = PowerThrottling(1, 1, 0)
        success = bool(kernel.SetProcessInformation(kernel.GetCurrentProcess(), 4,
                                                    ctypes.byref(state), ctypes.sizeof(state)))
        return {'high_qos': success, 'platform': 'nt', 'control_mask': 1,
                'state_mask': 0, 'error': 0 if success else ctypes.get_last_error()}
    except (AttributeError, OSError) as error:
        return {'high_qos': False, 'platform': 'nt', 'error': str(error)}
