"""跨帧保留占空比余额，逐请求仅排空ON包。"""
import ctypes
import json


class PhaseLoad:
    def __init__(self, library, log, cycle_ns, duty, phase):
        self.library = ctypes.CDLL(str(library))
        self.library.phase_load_create.argtypes = [ctypes.c_char_p,ctypes.c_ulonglong,ctypes.c_double,ctypes.c_double]
        self.library.phase_load_create.restype = ctypes.c_void_p
        for name in ('resume','pause','finish','information','destroy'):
            getattr(self.library,'phase_load_'+name).argtypes = [ctypes.c_void_p]
        self.library.phase_load_resume.restype = ctypes.c_long
        self.library.phase_load_pause.restype = ctypes.c_int
        self.library.phase_load_finish.restype = ctypes.c_char_p
        self.library.phase_load_information.restype = ctypes.c_char_p
        self.library.phase_load_error.restype = ctypes.c_char_p
        self.pointer = self.library.phase_load_create(str(log).encode(),cycle_ns,duty,phase)
        if not self.pointer:
            raise RuntimeError(self.library.phase_load_error().decode())
        self.duty = duty
        self.information = json.loads(self.library.phase_load_information(self.pointer).decode())

    def resume(self, stage, duty):
        assert duty == self.duty
        result = self.library.phase_load_resume(self.pointer)
        if result < 0:
            raise RuntimeError(self.library.phase_load_error().decode())
        return result

    def pause(self):
        if self.library.phase_load_pause(self.pointer):
            raise RuntimeError(self.library.phase_load_error().decode())

    def finish(self):
        result = self.library.phase_load_finish(self.pointer)
        if not result:
            raise RuntimeError(self.library.phase_load_error().decode())
        return json.loads(result.decode())

    def close(self):
        if self.pointer:
            self.library.phase_load_destroy(self.pointer)
            self.pointer = None
