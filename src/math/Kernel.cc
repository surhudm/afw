// -*- LSST-C++ -*-
/**
 * @file
 *
 * @brief Definitions of Kernel member functions.
 *
 * @ingroup afw
 */
#include <stdexcept>

#include "boost/format.hpp"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/Kernel.h"

lsst::afw::math::generic_kernel_tag lsst::afw::math::generic_kernel_tag_; ///< Used as default value in argument lists
lsst::afw::math::deltafunction_kernel_tag lsst::afw::math::deltafunction_kernel_tag_; ///< Used as default value in argument lists

//
// Constructors
//
/**
 * @brief Construct a spatially varying Kernel with one spatial function copied as needed
 *
 * @throw lsst::pex::exceptions::InvalidParameter if the kernel has no parameters.
 */
namespace {
}
lsst::afw::math::Kernel::Kernel(
    int width,                          ///< number of columns
    int height,                         ///< number of height
    unsigned int nKernelParams,         ///< number of kernel parameters
    SpatialFunction const &spatialFunction) ///< spatial function (or NullSpatialFunction if none is specified)
:
    LsstBase(typeid(this)),
    _width(width),
    _height(height),
    _ctrX((width-1)/2),
    _ctrY((height-1)/2),
    _nKernelParams(nKernelParams),
    _spatialFunctionList()
{
    if (dynamic_cast<const NullSpatialFunction*>(&spatialFunction)) {
        // spatialFunction is not really present
    } else {
        if (nKernelParams == 0) {
            throw lsst::pex::exceptions::InvalidParameter("Kernel function has no parameters");
        }
        for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
            SpatialFunctionPtr spatialFunctionCopy = spatialFunction.copy();
            this->_spatialFunctionList.push_back(spatialFunctionCopy);
        }
    }
}

/**
 * @brief Construct a spatially varying Kernel with a list of spatial functions (one per kernel parameter)
 *
 * Note: if the list of spatial functions is empty then the kernel is not spatially varying.
 */
lsst::afw::math::Kernel::Kernel(
    int width,                          ///< number of columns
    int height,                         ///< number of height
    std::vector<SpatialFunctionPtr> spatialFunctionList) ///< list of spatial function, one per kernel parameter
:
    LsstBase(typeid(this)),
   _width(width),
   _height(height),
   _ctrX((width-1)/2),
   _ctrY((height-1)/2),
   _nKernelParams(spatialFunctionList.size())
{
    for (unsigned int ii = 0; ii < spatialFunctionList.size(); ++ii) {
        SpatialFunctionPtr spatialFunctionCopy = spatialFunctionList[ii]->copy();
        this->_spatialFunctionList.push_back(spatialFunctionCopy);
    }
}

//
// Public Member Functions
//

/**
 * @brief Compute an image (pixellized representation of the kernel), returning a new image
 *
 * This would be called computeImage (overloading the other function of the same name)
 * but at least some versions of the g++ compiler cannot then reliably find the function.
 *
 * @return an image (your own copy to do with as you wish)
 */
lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT> lsst::afw::math::Kernel::computeNewImage(
    PixelT &imSum,  ///< sum of image pixels
    bool doNormalize,   ///< normalize the image (so sum is 1)?
    double x,   ///< x (column position) at which to compute spatial function
    double y    ///< y (row position) at which to compute spatial function
) const {
    lsst::afw::image::Image<lsst::afw::math::Kernel::PixelT> retImage(this->dimensions());
    this->computeImage(retImage, imSum, doNormalize, x, y);
    return retImage;
}

/**
 * @brief Set the parameters of all spatial functions
 *
 * Params is indexed as [kernel parameter][spatial parameter]
 *
 * @throw lsst::pex::exceptions::InvalidParameter if params is the wrong shape (and no parameters are changed)
 */
void lsst::afw::math::Kernel::setSpatialParameters(const std::vector<std::vector<double> > params) {
    // Check params size before changing anything
    unsigned int nKernelParams = this->getNKernelParameters();
    if (params.size() != nKernelParams) {
        throw lsst::pex::exceptions::InvalidParameter(
            boost::format("params has %d entries instead of %d") % params.size() % nKernelParams);
    }
    unsigned int nSpatialParams = this->getNSpatialParameters();
    for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
        if (params[ii].size() != nSpatialParams) {
            throw lsst::pex::exceptions::InvalidParameter(
                boost::format("params[%d] has %d entries instead of %d") %
                ii % params[ii].size() % nSpatialParams);
        }
    }
    // Set parameters
    for (unsigned int ii = 0; ii < nKernelParams; ++ii) {
        this->_spatialFunctionList[ii]->setParameters(params[ii]);
    }
}

/**
 * @brief Compute the kernel parameters at a specified point
 *
 * Warning: this is a low-level function that assumes kernelParams is the right length.
 * It will fail in unpredictable ways if that condition is not met.
 * The only reason it is not protected is because the convolveLinear function needs it.
 */
void lsst::afw::math::Kernel::computeKernelParametersFromSpatialModel(std::vector<double> &kernelParams, double x, double y) const {
    std::vector<double>::iterator paramIter = kernelParams.begin();
    std::vector<SpatialFunctionPtr>::const_iterator funcIter = _spatialFunctionList.begin();
    for ( ; funcIter != _spatialFunctionList.end(); ++funcIter, ++paramIter) {
        *paramIter = (*(*funcIter))(x,y);
    }
}

/**
 * @brief Return the current kernel parameters
 *
 * If the kernel is spatially varying then the parameters are those last computed.
 * See also computeKernelParametersFromSpatialModel.
 * If there are no kernel parameters then returns an empty vector.
 */
std::vector<double> lsst::afw::math::Kernel::getKernelParameters() const {
    return std::vector<double>();
}

/**
 * @brief Return a string representation of the kernel
 */
std::string lsst::afw::math::Kernel::toString(std::string prefix) const {
    std::ostringstream os;
    os << prefix << "Kernel:" << std::endl;
    os << prefix << "..height, width: " << _height << ", " << _width << std::endl;
    os << prefix << "..ctr (X, Y): " << _ctrX << ", " << _ctrY << std::endl;
    os << prefix << "..nKernelParams: " << _nKernelParams << std::endl;
    os << prefix << "..isSpatiallyVarying: " << (this->isSpatiallyVarying() ? "True" : "False") << std::endl;
    if (this->isSpatiallyVarying()) {
        os << prefix << "..spatialFunctions:" << std::endl;
        for (std::vector<SpatialFunctionPtr>::const_iterator spFuncPtr = _spatialFunctionList.begin();
            spFuncPtr != _spatialFunctionList.end(); ++spFuncPtr) {
            os << prefix << "...." << (*spFuncPtr)->toString() << std::endl;
        }
    }
    return os.str();
};

//
// Protected Member Functions
//

/**
 * @brief Set one kernel parameter
 *
 * Classes that have kernel parameters must subclass this function.
 *
 * This function is marked "const", despite modifying unimportant internals,
 * so that computeImage can be const.
 *
 * @throw lsst::pex::exceptions::RuntimeError always (unless subclassed)
 */
void lsst::afw::math::Kernel::setKernelParameter(int ind, double value) const {
    throw lsst::pex::exceptions::InvalidParameter("Kernel has no kernel parameters");
}

/**
 * @brief Set the kernel parameters from the spatial model (if any).
 *
 * This function has no effect if there is no spatial model.
 *
 * This function is marked "const", despite modifying unimportant internals,
 * so that computeImage can be const.
 */
void lsst::afw::math::Kernel::setKernelParametersFromSpatialModel(double x, double y) const {
    std::vector<SpatialFunctionPtr>::const_iterator funcIter = _spatialFunctionList.begin();
    for (int ii = 0; funcIter != _spatialFunctionList.end(); ++funcIter, ++ii) {
        this->setKernelParameter(ii, (*(*funcIter))(x,y));
    }
}
