/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <pybind11/pybind11.h>
#include <lsst/utils/python.h>

#include "lsst/afw/cameraGeom/Orientation.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace cameraGeom {

void wrapOrientation(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.addSignatureDependency("lsst.geom");
    wrappers.wrapType(py::class_<Orientation>(wrappers.module, "Orientation"), [](auto &mod, auto &cls) {
        /* Constructors */
        cls.def(py::init<lsst::geom::Point2D, lsst::geom::Point2D, lsst::geom::Angle, lsst::geom::Angle,
                         lsst::geom::Angle>(),
                "fpPosition"_a = lsst::geom::Point2D(0, 0), "refPoint"_a = lsst::geom::Point2D(-0.5, -0.5),
                "yaw"_a = lsst::geom::Angle(0), "pitch"_a = lsst::geom::Angle(0),
                "roll"_a = lsst::geom::Angle(0));

        /* Operators */

        /* Members */
        cls.def("getFpPosition", &Orientation::getFpPosition);
        cls.def("getReferencePoint", &Orientation::getReferencePoint);
        cls.def("getYaw", &Orientation::getYaw);
        cls.def("getPitch", &Orientation::getPitch);
        cls.def("getRoll", &Orientation::getRoll);
        cls.def("getNQuarter", &Orientation::getNQuarter);
        cls.def("makePixelFpTransform", &Orientation::makePixelFpTransform, "pixelSizeMm"_a);
        cls.def("makeFpPixelTransform", &Orientation::makeFpPixelTransform, "pixelSizeMm"_a);
        cls.def("getFpPosition", &Orientation::getFpPosition);
        cls.def("getFpPosition", &Orientation::getFpPosition);
        cls.def("getFpPosition", &Orientation::getFpPosition);
        cls.def("getFpPosition", &Orientation::getFpPosition);
    });
}
}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst
