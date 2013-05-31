// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2013 LSST Corporation.
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

%define ellipsesLib_DOCSTRING
"
Python interface to lsst::afw::geom::ellipses classes and functions
"
%enddef

%feature("autodoc", "1");
%module(package="lsst.afw.geom.ellipses", docstring=ellipsesLib_DOCSTRING) ellipsesLib

%{
#include "lsst/afw/geom.h"
#include "lsst/afw/geom/ellipses.h"
#define PY_ARRAY_UNIQUE_SYMBOL LSST_AFW_GEOM_ELLIPSES_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#include "ndarray/swig/eigen.h"
%}

%init%{
    import_array();
%}

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

%include "ndarray.i"
%import "lsst/afw/geom/geomLib.i"

%pythondynamic lsst::afw::geom::ellipses::Ellipse;

%ignore lsst::afw::geom::ellipses::EllipseCore::transform;
%ignore lsst::afw::geom::ellipses::EllipseCore::convolve;
%ignore lsst::afw::geom::ellipses::EllipseCore::getGridTransform;
%ignore lsst::afw::geom::ellipses::EllipseCore::readParameters;
%ignore lsst::afw::geom::ellipses::EllipseCore::writeParameters;
%ignore lsst::afw::geom::ellipses::EllipseCore::as;
%rename(assign) lsst::afw::geom::ellipses::EllipseCore::operator=;

%declareNumPyConverters(lsst::afw::geom::ellipses::Quadrupole::Matrix);
%declareNumPyConverters(lsst::afw::geom::ellipses::EllipseCore::Jacobian);
%declareNumPyConverters(lsst::afw::geom::ellipses::EllipseCore::ParameterVector);
%declareNumPyConverters(lsst::afw::geom::ellipses::EllipseCore::Transformer::DerivativeMatrix);
%declareNumPyConverters(lsst::afw::geom::ellipses::EllipseCore::Transformer::TransformDerivativeMatrix);
%declareNumPyConverters(lsst::afw::geom::ellipses::EllipseCore::GridTransform::DerivativeMatrix);
%declareNumPyConverters(lsst::afw::geom::ellipses::EllipseCore::Convolution::DerivativeMatrix);

%shared_ptr(lsst::afw::geom::ellipses::EllipseCore);

%include "lsst/afw/geom/ellipses/EllipseCore.h"

%addStreamRepr(lsst::afw::geom::ellipses::EllipseCore);
%extend lsst::afw::geom::ellipses::EllipseCore {
    %pythoncode %{
        def transform(self, transform, inPlace=False, doDerivatives=False):
            """Transform an EllipseCore via a LinearTransform

            If inPlace, self will be transformed in-place and returned;
            otherwise, a new transformed EllipseCore of the same type
            will be returned.

            If doDerivatives, a tuple of (transformed, dEllipse, dTransform) will be
            returned, where dEllipse is the derivative of the transformed ellipse
            w.r.t. the input ellipse, and dTransform is the derivative of the
            transformed ellipse w.r.t. the LinearTransform elements.
            """
            if doDerivatives:
                dEllipse = _ellipsesLib.EllipseCore__transformDEllipse(self, transform)
                dTransform = _ellipsesLib.EllipseCore__transformDTransform(self, transform)
            if inPlace:
                r = self
            else:
                r = self.clone()
            _ellipsesLib.EllipseCore__transformInPlace(r, transform)
            if doDerivatives:
                return r, dEllipse, dTransform
            else:
                return r
    %}
    %feature("shadow") _transformInPlace %{%}
    void _transformInPlace(lsst::afw::geom::LinearTransform const & t) {
       self->transform(t).inPlace();
    }
    %feature("shadow") _transformDEllipse %{%}
    lsst::afw::geom::ellipses::EllipseCore::Transformer::DerivativeMatrix _transformDEllipse(
        lsst::afw::geom::LinearTransform const & t
    ) const {
        return self->transform(t).d();
    }
    %feature("shadow") _transformDTransform %{%}
    lsst::afw::geom::ellipses::EllipseCore::Transformer::TransformDerivativeMatrix _transformDTransform(
        lsst::afw::geom::LinearTransform const & t
    ) const {
        return self->transform(t).dTransform();
    }

    %feature("shadow") _getGridTransform %{
        def getGridTransform(self, doDerivatives=False):
            """Return the LinearTransform that maps self to the unit circle

            If doDerivatives, a tuple of (transform, derivative) will be
            returned, where 'derivative' is the derivative of the LinearTransform
            w.r.t. the input ellipse.
            """
            if doDerivatives:
                return $action(self), $actionD(self)
            return $action(self)
    %}
    lsst::afw::geom::LinearTransform _getGridTransform() {
        return self->getGridTransform();
    }
    %feature("shadow") _getGridTransformD %{%}
    lsst::afw::geom::ellipses::EllipseCore::GridTransform::DerivativeMatrix _getGridTransformD() {
        return self->getGridTransform().d();
    }

    %pythoncode %{
        def convolve(self, other, inPlace=False, doDerivatives=False):
            """Given the 2nd-moment ellipses of two images, compute the 2nd-moment
            ellipse of the convolution of the images.

            If inPlace, self will be convolved in-place and returned;
            otherwise, a new convolved EllipseCore of the same type
            will be returned.

            If doDerivatives, a tuple of (convolved, derivative) will be returned,
            where 'derivative' is the derivative of the convolved EllipseCore w.r.t.
            'self'.
            """
            if doDerivatives:
                derivative = _ellipsesLib.EllipseCore__convolveD(self, other)
            if inPlace:
                r = self
            else:
                r = self.clone()
            _ellipsesLib.EllipseCore__convolveInPlace(r, other)
            if doDerivatives:
                return r, derivative
            else:
                return r
    %}
    %feature("shadow") _convolveInPlace %{%}
    void _convolveInPlace(lsst::afw::geom::ellipses::EllipseCore const & other) {
       self->convolve(other).inPlace();
    }
    %feature("shadow") _convolveD %{%}
    lsst::afw::geom::ellipses::EllipseCore::Convolution::DerivativeMatrix _convolveD(
        lsst::afw::geom::ellipses::EllipseCore const & other
    ) const {
        return self->convolve(other).d();
    }

    %feature("shadow") as_ %{
        def as_(self, cls):
            """Return a new EllipseCore equivalent to self, with type specified
            by the given string name or type object.
            """
            if isinstance(cls, basestring):
                return $action(self, cls)
            return cls(self)
    %}
    PTR(lsst::afw::geom::ellipses::EllipseCore) as_(std::string const & name) {
        return self->as(name);
    }

    %pythoncode %{
        def __reduce__(self):
            return (EllipseCore.make, (name, self.getParameterVector()))
    %}
}

%define %EllipseCore_PREINCLUDE(NAME)
%feature(notabstract) lsst::afw::geom::ellipses::NAME;
%implicitconv lsst::afw::geom::ellipses::NAME;
%shared_ptr(lsst::afw::geom::ellipses::NAME);
%rename(assign) lsst::afw::geom::ellipses::NAME::operator=;
%enddef

%define %EllipseCore_POSTINCLUDE(NAME)
%extend lsst::afw::geom::ellipses::NAME {

    static PTR(lsst::afw::geom::ellipses::NAME) cast(
        PTR(lsst::afw::geom::ellipses::EllipseCore) const & p
    ) {
       return boost::dynamic_pointer_cast<lsst::afw::geom::ellipses::NAME>(p);
    }
}
%enddef

%EllipseCore_PREINCLUDE(Axes);
%EllipseCore_PREINCLUDE(Quadrupole);

%include "lsst/afw/geom/ellipses/Axes.h"
%include "lsst/afw/geom/ellipses/Quadrupole.h"

%EllipseCore_POSTINCLUDE(Axes);
%EllipseCore_POSTINCLUDE(Quadrupole);

%include "Separable.i"

%feature("valuewrapper") lsst::afw::geom::ellipses::Ellipse;
%ignore lsst::afw::geom::ellipses::Ellipse::getCore;
%ignore lsst::afw::geom::ellipses::Ellipse::transform;
%ignore lsst::afw::geom::ellipses::Ellipse::convolve;
%ignore lsst::afw::geom::ellipses::Ellipse::getGridTransform;
%ignore lsst::afw::geom::ellipses::Ellipse::readParameters;
%ignore lsst::afw::geom::ellipses::Ellipse::writeParameters;

%rename(assign) lsst::afw::geom::ellipses::Ellipse::operator=;

%shared_ptr(lsst::afw::geom::ellipses::Ellipse);
%declareNumPyConverters(lsst::afw::geom::ellipses::Ellipse::ParameterVector);
%declareNumPyConverters(lsst::afw::geom::ellipses::Ellipse::Transformer::DerivativeMatrix);
%declareNumPyConverters(lsst::afw::geom::ellipses::Ellipse::Transformer::TransformDerivativeMatrix);
%declareNumPyConverters(lsst::afw::geom::ellipses::Ellipse::GridTransform::DerivativeMatrix);
%declareNumPyConverters(lsst::afw::geom::ellipses::Ellipse::Convolution::DerivativeMatrix);

%addStreamRepr(lsst::afw::geom::ellipses::Ellipse);

%extend lsst::afw::geom::ellipses::Ellipse {
    %pythoncode %{
        def transform(self, transform, inPlace=False, doDerivatives=False):
            """Transform an Ellipse via an AffineTransform

            If inPlace, self will be transformed in-place and returned;
            otherwise, a new transformed ellipse with the same EllipseCore type
            will be returned.

            If doDerivatives, a tuple of (transformed, dEllipse, dTransform) will be
            returned, where dEllipse is the derivative of the transformed ellipse
            w.r.t. the input ellipse, and dTransform is the derivative of the
            transformed ellipse w.r.t. the AffineTransform elements.
            """
            if doDerivatives:
                dEllipse = _ellipsesLib.Ellipse__transformDEllipse(self, transform)
                dTransform = _ellipsesLib.Ellipse__transformDTransform(self, transform)
            if inPlace:
                r = self
            else:
                r = Ellipse(self)
            _ellipsesLib.Ellipse__transformInPlace(r, transform)
            if doDerivatives:
                return r, dEllipse, dTransform
            else:
                return r
    %}
    %feature("shadow") _transformInPlace %{%}
    void _transformInPlace(lsst::afw::geom::AffineTransform const & t) {
        self->transform(t).inPlace();
    }
    %feature("shadow") _transformDEllipse %{%}
    lsst::afw::geom::ellipses::Ellipse::Transformer::DerivativeMatrix _transformDEllipse(
        lsst::afw::geom::AffineTransform const & t
    ) const {
        return self->transform(t).d();
    }
    %feature("shadow") _transformDTransform %{%}
    lsst::afw::geom::ellipses::Ellipse::Transformer::TransformDerivativeMatrix _transformDTransform(
        lsst::afw::geom::AffineTransform const & t
    ) const {
        return self->transform(t).dTransform();
    }

    %feature("shadow") _getGridTransform %{
        def getGridTransform(self, doDerivatives=False):
            """Return the AffineTransform that maps self to the unit circle at the origin

            If doDerivatives, a tuple of (transform, derivative) will be
            returned, where 'derivative' is the derivative of the LinearTransform
            w.r.t. the input ellipse.
            """
            if doDerivatives:
                return $action(self), $actionD(self)
            return $action(self)
    %}
    lsst::afw::geom::AffineTransform _getGridTransform() {
        return self->getGridTransform();
    }
    %feature("shadow") _getGridTransformD %{%}
    lsst::afw::geom::ellipses::Ellipse::GridTransform::DerivativeMatrix _getGridTransformD() {
        return self->getGridTransform().d();
    }

    %pythoncode %{
        def convolve(self, other, inPlace=False, doDerivatives=False):
            """Given the 2nd-moment ellipses of two images, compute the 2nd-moment
            ellipse of the convolution of the images.

            If inPlace, self will be convolved in-place and returned;
            otherwise, a new convolved Ellipse of the same type
            will be returned.

            If doDerivatives, a tuple of (convolved, derivative) will be returned,
            where 'derivative' is the derivative of the convolved Ellipse w.r.t.
            'self'.
            """
            if doDerivatives:
                derivative = _ellipsesLib.Ellipse__convolveD(self, other)
            if inPlace:
                r = self
            else:
                r = Ellipse(self)
            _ellipsesLib.Ellipse__convolveInPlace(r, other)
            if doDerivatives:
                return r, derivative
            else:
                return r
    %}
    %feature("shadow") _convolveInPlace %{%}
    void _convolveInPlace(lsst::afw::geom::ellipses::Ellipse const & other) {
       self->convolve(other).inPlace();
    }
    %feature("shadow") _convolveD %{%}
    lsst::afw::geom::ellipses::Ellipse::Convolution::DerivativeMatrix _convolveD(
        lsst::afw::geom::ellipses::Ellipse const & other
    ) const {
        return self->convolve(other).d();
    }

    %feature("shadow") getCorePtr %{
        def getCore(self):
            return $action(self).cast()
    %}

    %pythoncode %{
        def __reduce__(self):
            return (Ellipse, (self.getCore(), self.getCenter()))
    %}
}

%include "lsst/afw/geom/ellipses/Ellipse.h"
%include "lsst/afw/geom/ellipses/Parametric.h"
%include "lsst/afw/geom/ellipses.h" // just for Separable typedefs

%include "PixelRegion.i"
