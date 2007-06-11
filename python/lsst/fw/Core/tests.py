"""Support code for running unit tests"""

import unittest
import lsst.fw.Core.fwLib as fw

try:
    type(memId0)
except NameError:
    memId0 = 0

def init():
    global memId0
    memId0 = fw.Citizen_getNextMemId()  # used by MemoryTestCase

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        
class MemoryTestCase(unittest.TestCase):
    """Check for memory leaks since memId0 was allocated"""
    def setUp(self):
        pass

    def testLeaks(self):
        """Check for memory leaks in the preceding tests"""

        global memId0
        if fw.Citizen_census(0, memId0) != 0:
            if not False:
                print fw.Citizen_census(0, memId0), "Objects leaked:"
                print fw.Citizen_census(fw.cout, memId0)
                
            self.fail("Leaked %d blocks" % fw.Citizen_census(0, memId0))
