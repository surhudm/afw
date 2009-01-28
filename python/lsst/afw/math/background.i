
%{
#include "lsst/afw/math/Background.h"
%}

%include "lsst/afw/math/Background.h"

%define %declareBack(PIXTYPE, SUFFIX)
    %template(make_Background) lsst::afw::math::make_Background<lsst::afw::image::Image<PIXTYPE> >;
    %template(Background ## SUFFIX) lsst::afw::math::Background::Background<lsst::afw::image::Image<PIXTYPE> >;
    %template(getImage ## SUFFIX) lsst::afw::math::Background::getImage<PIXTYPE>;
%enddef

%declareBack(double, D)
%declareBack(float, F)
%declareBack(int, I)


