# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Tests for PCA on Images

Run with:
   python test_imagePca.py
or
   pytest test_imagePca.py
"""

import unittest
import random
import math
import itertools

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions as pexExcept
import lsst.geom
import lsst.afw.image as afwImage
import lsst.afw.display as afwDisplay

try:
    type(display)
except NameError:
    display = False


class ImagePcaTestCase(lsst.utils.tests.TestCase):
    """A test case for ImagePca"""

    def setUp(self):
        random.seed(0)
        self.ImageSet = afwImage.ImagePcaF()

    def tearDown(self):
        del self.ImageSet

    def testInnerProducts(self):
        """Test inner products"""

        width, height = 10, 20
        im1 = afwImage.ImageF(lsst.geom.Extent2I(width, height))
        val1 = 10
        im1.set(val1)

        im2 = im1.Factory(im1.getDimensions())
        val2 = 20
        im2.set(val2)

        self.assertEqual(afwImage.innerProduct(im1, im1),
                         width*height*val1*val1)
        self.assertEqual(afwImage.innerProduct(im1, im2),
                         width*height*val1*val2)

        im2[0, 0, afwImage.LOCAL] = 0
        self.assertEqual(afwImage.innerProduct(im1, im2),
                         (width*height - 1)*val1*val2)

        im2[0, 0, afwImage.LOCAL] = val2             # reinstate value
        im2[width - 1, height - 1, afwImage.LOCAL] = 1
        self.assertEqual(afwImage.innerProduct(im1, im2),
                         (width*height - 1)*val1*val2 + val1)

    def testAddImages(self):
        """Test adding images to a PCA set"""

        nImage = 3
        for i in range(nImage):
            im = afwImage.ImageF(lsst.geom.Extent2I(21, 21))
            val = 1
            im.set(val)

            self.ImageSet.addImage(im, 1.0)

        vec = self.ImageSet.getImageList()
        self.assertEqual(len(vec), nImage)
        self.assertEqual(vec[nImage - 1][0, 0, afwImage.LOCAL], val)

        def tst():
            """Try adding an image with no flux"""
            self.ImageSet.addImage(im, 0.0)

        self.assertRaises(pexExcept.OutOfRangeError, tst)

    def testMean(self):
        """Test calculating mean image"""

        width, height = 10, 20

        values = (100, 200, 300)
        meanVal = 0
        for val in values:
            im = afwImage.ImageF(lsst.geom.Extent2I(width, height))
            im.set(val)

            self.ImageSet.addImage(im, 1.0)
            meanVal += val

        meanVal = meanVal/len(values)

        mean = self.ImageSet.getMean()

        self.assertEqual(mean.getWidth(), width)
        self.assertEqual(mean.getHeight(), height)
        self.assertEqual(mean[0, 0, afwImage.LOCAL], meanVal)
        self.assertEqual(mean[width - 1, height - 1, afwImage.LOCAL], meanVal)

    def testPca(self):
        """Test calculating PCA"""
        width, height = 200, 100
        numBases = 3
        numInputs = 3

        bases = []
        for i in range(numBases):
            im = afwImage.ImageF(width, height)
            array = im.getArray()
            x, y = np.indices(array.shape)
            period = 5*(i+1)
            fx = np.sin(2*math.pi/period*x + 2*math.pi/numBases*i)
            fy = np.sin(2*math.pi/period*y + 2*math.pi/numBases*i)
            array[x, y] = fx + fy
            bases.append(im)

        if display:
            mos = afwDisplay.utils.Mosaic(background=-10)
            afwDisplay.Display(frame=1).mtv(mos.makeMosaic(bases), title="Basis functions")

        inputs = []
        for i in range(numInputs):
            im = afwImage.ImageF(lsst.geom.Extent2I(width, height))
            im.set(0)
            for b in bases:
                im.scaledPlus(random.random(), b)

            inputs.append(im)
            self.ImageSet.addImage(im, 1.0)

        if display:
            mos = afwDisplay.utils.Mosaic(background=-10)
            afwDisplay.Display(frame=2).mtv(mos.makeMosaic(inputs), title="Inputs")

        self.ImageSet.analyze()

        eImages = []
        for img in self.ImageSet.getEigenImages():
            eImages.append(img)

        if display:
            mos = afwDisplay.utils.Mosaic(background=-10)
            afwDisplay.Display(frame=3).mtv(mos.makeMosaic(eImages), title="EigenImages")

        self.assertEqual(len(eImages), numInputs)

        # Test for orthogonality
        for i1, i2 in itertools.combinations(list(range(len(eImages))), 2):
            inner = afwImage.innerProduct(eImages[i1], eImages[i2])
            norm1 = eImages[i1].getArray().sum()
            norm2 = eImages[i2].getArray().sum()
            inner /= norm1*norm2
            self.assertAlmostEqual(inner, 0, 6)

    def testPcaNaN(self):
        """Test calculating PCA when the images can contain NaNs"""

        width, height = 20, 10

        values = (100, 200, 300)
        for i, val in enumerate(values):
            im = afwImage.ImageF(lsst.geom.Extent2I(width, height))
            im.set(val)

            if i == 1:
                im[width//2, height//2, afwImage.LOCAL] = np.nan

            self.ImageSet.addImage(im, 1.0)

        self.ImageSet.analyze()

        eImages = []
        for img in self.ImageSet.getEigenImages():
            eImages.append(img)

        if display:
            mos = afwDisplay.utils.Mosaic(background=-10)
            afwDisplay.Display(frame=0).mtv(mos.makeMosaic(eImages), title="testPcaNaN")


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
