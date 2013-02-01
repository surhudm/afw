// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
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
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
 
%define mathLib_DOCSTRING
"
Python interface to lsst::afw::math classes
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.math",docstring=mathLib_DOCSTRING) mathLib

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

namespace lsst { namespace afw { namespace math {

class Kernel;
class AnalyticKernel;
class DeltaFunctionKernel;
class FixedKernel;
class LinearCombinationKernel;
class SeparableKernel;
class Background;
class BackgroundMI;
class StatisticsControl;
class Statistics;
class Random;
class LeastSquares;
class CandidateVisistor;
class SpatialCellCandidate;
class SpatialCell;

}}} // namespace lsst::afw::math

%shared_ptr(lsst::afw::math::Kernel);
%shared_ptr(lsst::afw::math::AnalyticKernel);
%shared_ptr(lsst::afw::math::DeltaFunctionKernel);
%shared_ptr(lsst::afw::math::FixedKernel);
%shared_ptr(lsst::afw::math::LinearCombinationKernel);
%shared_ptr(lsst::afw::math::SeparableKernel);
%shared_ptr(lsst::afw::math::Background);
%shared_ptr(lsst::afw::math::BackgroundMI);
%shared_ptr(lsst::afw::math::StatisticsControl);
%shared_ptr(lsst::afw::math::CandidateVisitor)
%shared_ptr(lsst::afw::math::SpatialCellCandidate);
%shared_ptr(lsst::afw::math::SpatialCell);
