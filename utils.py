from enum import Enum


class Factor(Enum):
    BETA = 'Mkt-RF'
    SMB = 'SMB'
    HML = 'HML'
    RMW = 'RMW'
    CMA = 'CMA'
    MOM = 'WML'
    MKT = 'Mkt'
    RF = 'RF'
    FX = 'FX'
    USRF = 'USRF'
    USBETA = 'USMkt-USRF'
    USMKT = 'USMkt'

    def __get__(self, instance, owner):
        return self.value
