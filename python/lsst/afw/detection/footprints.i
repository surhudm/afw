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
 
%{
#include "lsst/afw/detection/Threshold.h"
#include "lsst/afw/detection/Peak.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/FootprintSet.h"
#include "lsst/afw/detection/FootprintFunctor.h"
#include "lsst/afw/detection/FootprintArray.h"
#include "lsst/afw/detection/FootprintArray.cc"
%}

// std_vector.i is broken when using %shared_ptr(std::vector<...>)
// apparently because %typemap_traits_ptr() overwrites typemaps setup
// by %shared_ptr. Therefore, create a std::vector specialization visible
// only to swig for a specific type, and move the %typemap_traits_ptr()
// invocation post-vector-method expansion.
%define %shared_vec(TYPE...)
    namespace std {
        template <class _Alloc >
        class vector<TYPE, _Alloc > {
        public:
            typedef size_t size_type;
            typedef ptrdiff_t difference_type;
            typedef TYPE value_type;
            typedef value_type* pointer;
            typedef const value_type* const_pointer;
            typedef TYPE& reference;
            typedef const TYPE& const_reference;
            typedef _Alloc allocator_type;

            %traits_swigtype(TYPE);

            %fragment(SWIG_Traits_frag(std::vector<TYPE, _Alloc >), "header",
                      fragment=SWIG_Traits_frag(TYPE),
                      fragment="StdVectorTraits") {
                namespace swig {
                    template <>  struct traits<std::vector<TYPE, _Alloc > > {
                        typedef pointer_category category;
                        static const char* type_name() {
                            return "std::vector<" #TYPE " >";
                        }
                    };
                }
            }

            %swig_vector_methods(std::vector<TYPE, _Alloc >);
            %std_vector_methods(vector);

            %typemap_traits_ptr(SWIG_TYPECHECK_VECTOR, std::vector<TYPE, _Alloc >);
        };
    }
%enddef

%shared_vec(boost::shared_ptr<lsst::afw::detection::Footprint>);


%ignore lsst::afw::detection::FootprintFunctor::operator();

// already in image.i.
// %template(VectorBox2I) std::vector<lsst::afw::geom::Box2I>;

%shared_ptr(lsst::afw::detection::Peak);
%shared_ptr(lsst::afw::detection::Footprint);
%shared_ptr(lsst::afw::detection::Span);
%shared_ptr(lsst::afw::detection::FootprintSet<boost::uint16_t, lsst::afw::image::MaskPixel>);
%shared_ptr(lsst::afw::detection::FootprintSet<int, lsst::afw::image::MaskPixel>);
%shared_ptr(lsst::afw::detection::FootprintSet<float, lsst::afw::image::MaskPixel>);
%shared_ptr(lsst::afw::detection::FootprintSet<double, lsst::afw::image::MaskPixel>);
%shared_ptr(std::vector<boost::shared_ptr<lsst::afw::detection::Footprint> >);

%rename(assign) lsst::afw::detection::Footprint::operator=;

%include "lsst/afw/detection/Threshold.h"
%include "lsst/afw/detection/Peak.h"
%include "lsst/afw/detection/Footprint.h"
%include "lsst/afw/detection/FootprintSet.h"
%include "lsst/afw/detection/FootprintFunctor.h"

%define %thresholdOperations(TYPE)
    %extend lsst::afw::detection::Threshold {
        %template(getValue) getValue<TYPE<unsigned short> >;
        %template(getValue) getValue<TYPE<int> >;
        %template(getValue) getValue<TYPE<float> >;
        %template(getValue) getValue<TYPE<double> >;
    }
%enddef

%define %footprintOperations(PIXEL)
%template(insertIntoImage) lsst::afw::detection::Footprint::insertIntoImage<PIXEL>;
%enddef

%extend lsst::afw::detection::Footprint {
    %template(intersectMask) intersectMask<lsst::afw::image::MaskPixel>;
    %footprintOperations(unsigned short)
    %footprintOperations(int)
    %footprintOperations(boost::uint64_t)
}

%template(PeakContainerT)      std::vector<boost::shared_ptr<lsst::afw::detection::Peak> >;
%template(SpanContainerT)      std::vector<boost::shared_ptr<lsst::afw::detection::Span> >;
%template(FootprintList)       std::vector<boost::shared_ptr<lsst::afw::detection::Footprint> >;

%define %imageOperations(NAME, PIXEL_TYPE)
    %template(FootprintFunctor ##NAME) lsst::afw::detection::FootprintFunctor<lsst::afw::image::Image<PIXEL_TYPE> >;
    %template(FootprintFunctorMI ##NAME)
                       lsst::afw::detection::FootprintFunctor<lsst::afw::image::MaskedImage<PIXEL_TYPE> >;
    %template(setImageFromFootprint) lsst::afw::detection::setImageFromFootprint<lsst::afw::image::Image<PIXEL_TYPE> >;
    %template(setImageFromFootprintList)
                       lsst::afw::detection::setImageFromFootprintList<lsst::afw::image::Image<PIXEL_TYPE> >
%enddef

%define %maskOperations(PIXEL_TYPE)
    %template(footprintAndMask) lsst::afw::detection::footprintAndMask<PIXEL_TYPE>;
    %template(setMaskFromFootprint) lsst::afw::detection::setMaskFromFootprint<PIXEL_TYPE>;
    %template(setMaskFromFootprintList) lsst::afw::detection::setMaskFromFootprintList<PIXEL_TYPE>;
%enddef

%define %FootprintSet(NAME, PIXEL_TYPE)
%template(FootprintSet##NAME) lsst::afw::detection::FootprintSet<PIXEL_TYPE, lsst::afw::image::MaskPixel>;
%template(makeFootprintSet) lsst::afw::detection::makeFootprintSet<PIXEL_TYPE, lsst::afw::image::MaskPixel>;
%enddef

%thresholdOperations(lsst::afw::image::Image);
%thresholdOperations(lsst::afw::image::MaskedImage);
%imageOperations(F, float);
%imageOperations(D, double);
%maskOperations(lsst::afw::image::MaskPixel);
%template(FootprintFunctorMaskU) lsst::afw::detection::FootprintFunctor<lsst::afw::image::Mask<boost::uint16_t> >;

%FootprintSet(U, boost::uint16_t);
%FootprintSet(I, int);
%FootprintSet(D, double);
%FootprintSet(F, float);
%template(makeFootprintSet) lsst::afw::detection::makeFootprintSet<lsst::afw::image::MaskPixel>;

%extend lsst::afw::detection::Span {
    %pythoncode {
    def __str__(self):
        """Print this Span"""
        return self.toString()
    }
}

// because stupid SWIG's %template doesn't work on these functions
%define %footprintArrayTemplates(T)
%declareNumPyConverters(lsst::ndarray::Array<T,1,0>);
%declareNumPyConverters(lsst::ndarray::Array<T,2,0>);
%declareNumPyConverters(lsst::ndarray::Array<T,3,0>);
%declareNumPyConverters(lsst::ndarray::Array<T const,1,0>);
%declareNumPyConverters(lsst::ndarray::Array<T const,2,0>);
%declareNumPyConverters(lsst::ndarray::Array<T const,3,0>);
%inline %{
    void flattenArray(
        lsst::afw::detection::Footprint const & fp,
        lsst::ndarray::Array<T const,2,0> const & src,
        lsst::ndarray::Array<T,1,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::flattenArray(fp, src, dest, origin);
    }    
    void flattenArray(
        lsst::afw::detection::Footprint const & fp,
        lsst::ndarray::Array<T const,3,0> const & src,
        lsst::ndarray::Array<T,2,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::flattenArray(fp, src, dest, origin);
    }    
    void expandArray(
        lsst::afw::detection::Footprint const & fp,
        lsst::ndarray::Array<T const,1,0> const & src,
        lsst::ndarray::Array<T,2,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::expandArray(fp, src, dest, origin);
    }
    void expandArray(
        lsst::afw::detection::Footprint const & fp,
        lsst::ndarray::Array<T const,2,0> const & src,
        lsst::ndarray::Array<T,3,0> const & dest,
        lsst::afw::geom::Point2I const & origin = lsst::afw::geom::Point2I()
    ) {
        lsst::afw::detection::expandArray(fp, src, dest, origin);
    }
%}
%{
    template void lsst::afw::detection::flattenArray(
        lsst::afw::detection::Footprint const &,
        lsst::ndarray::Array<T const,2,0> const &,
        lsst::ndarray::Array<T,1,0> const &,
        lsst::afw::geom::Point2I const &
    );
    template void lsst::afw::detection::flattenArray(
        lsst::afw::detection::Footprint const &,
        lsst::ndarray::Array<T const,3,0> const &,
        lsst::ndarray::Array<T,2,0> const &,
        lsst::afw::geom::Point2I const &
    );
    template void lsst::afw::detection::expandArray(
        lsst::afw::detection::Footprint const &,
        lsst::ndarray::Array<T const,1,0> const &,
        lsst::ndarray::Array<T,2,0> const &,
        lsst::afw::geom::Point2I const &
    );
    template void lsst::afw::detection::expandArray(
        lsst::afw::detection::Footprint const &,
        lsst::ndarray::Array<T const,2,0> const &,
        lsst::ndarray::Array<T,3,0> const &,
        lsst::afw::geom::Point2I const &
    );
%}
%enddef

%footprintArrayTemplates(boost::uint16_t);
%footprintArrayTemplates(int);
%footprintArrayTemplates(float);
%footprintArrayTemplates(double);
