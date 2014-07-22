changecom(`###')dnl
// -*- lsst-c++ -*-
/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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

// THIS FILE IS AUTOMATICALLY GENERATED from Source.h.m4, AND WILL BE OVERWRITTEN IF EDITED MANUALLY.

define(`m4def', defn(`define'))dnl
m4def(`DECLARE_SLOT_GETTERS',
`/// @brief Get the value of the $1$2 slot measurement.
    $2::MeasValue get$1$2() const;

    /// @brief Get the uncertainty on the $1$2 slot measurement.
    $2::ErrValue get$1$2Err() const;

    /// @brief Return true if the measurement in the $1$2 slot was successful.
    bool get$1$2Flag() const;
')dnl
m4def(`DECLARE_FLUX_GETTERS', `DECLARE_SLOT_GETTERS($1, `Flux')')dnl
m4def(`DECLARE_CENTROID_GETTERS', `DECLARE_SLOT_GETTERS(`', `Centroid')')dnl
m4def(`DECLARE_SHAPE_GETTERS', `DECLARE_SLOT_GETTERS(`', `Shape')')dnl
m4def(`DEFINE_SLOT_GETTERS',
`inline $2::MeasValue SourceRecord::get$1$2() const {
    return this->get(getTable()->get$1$2Key());
}

inline $2::ErrValue SourceRecord::get$1$2Err() const {
    return this->get(getTable()->get$1$2ErrKey());
}

inline bool SourceRecord::get$1$2Flag() const {
    return this->get(getTable()->get$1$2FlagKey());
}
')dnl
m4def(`DEFINE_FLUX_GETTERS', `DEFINE_SLOT_GETTERS($1, `Flux')')dnl
m4def(`DEFINE_CENTROID_GETTERS', `DEFINE_SLOT_GETTERS(`', `Centroid')')dnl
m4def(`DEFINE_SHAPE_GETTERS', `DEFINE_SLOT_GETTERS(`', `Shape')')dnl

// with the current state of the slots, this macro is only used for flux slots - pgee
m4def(`DECLARE_SLOT_DEFINERS',
`
    /**
     *  @brief Set the measurement used for the $1$2 slot with a field name.
     *
     *  For version 0 tables, requires that the measurement adhere to the convention
     *  of having "<name>", "<name>.err", and "<name>.flags" fields for all three fields
     *  to be attached to slots.  Only the main measurement field is required.
     */
    void define$1$2(std::string const & name) {
        Schema schema = getSchema();
        _slot$2$3.name = name;
        if (getVersion() == 0) {
            _slot$2$3.$4 = schema[name];
            try {
                _slot$2$3.$4Sigma = schema[name]["err"];
            } catch (pex::exceptions::NotFoundError) {}
            try {
                _slot$2$3.flag = schema[name]["flags"];
            } catch (pex::exceptions::NotFoundError) {}
            return;
        }
        _slot$2$3.$4 = schema[name + "_$4"];
        try {
            _slot$2$3.$4Sigma = schema[name + "_$4Sigma"];
        } catch (pex::exceptions::NotFoundError) {}
        try {
            _slot$2$3.flag = schema[name + "_flag"];
        } catch (pex::exceptions::NotFoundError) {}
    }

    /// @brief Return the name of the field used for the $1$2 slot.
    std::string get$1$2Definition() const {
        return _slot$2$3.name;
    }
    /// @brief Return the true if the Centroid slot is valid

    bool has$1$2Slot() const {
        return _slot$2$3.flux.isValid();
    }

    /// @brief Return the key used for the $1$2 slot.
    $2::MeasKey get$1$2Key() const { 
        return _slot$2$3.$4;
    }

    /// @brief Return the key used for $1$2 slot error or covariance.
    $2::ErrKey get$1$2ErrKey() const {
        return _slot$2$3.$4Sigma;
    }

    /// @brief Return the key used for the $1$2 slot success flag.
    Key<Flag> get$1$2FlagKey() const {
        return _slot$2$3.flag;
    }
')dnl
m4def(`DECLARE_FLUX_DEFINERS', `DECLARE_SLOT_DEFINERS($1, `Flux', `[FLUX_SLOT_`'translit($1, `a-z', `A-Z')]', `flux')')dnl
define(`m4def', defn(`define'))dnl
m4def(`DEFINE_FLUX_COLUMN_GETTERS',
`/// @brief Get the value of the $1Flux slot measurement.
    ndarray::Array<double,1> get$1Flux() const {
        return this->operator[](this->getTable()->get$1FluxKey());
    }
    /// @brief Get the uncertainty on the $1Flux slot measurement.
    ndarray::Array<double,1> get$1FluxErr() const {
        return this->operator[](this->getTable()->get$1FluxErrKey());
    }
')dnl
#ifndef AFW_TABLE_Source_h_INCLUDED
#define AFW_TABLE_Source_h_INCLUDED

#include "boost/array.hpp"
#include "boost/type_traits/is_convertible.hpp"

#include "lsst/utils/ieee.h"
#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/table/Simple.h"
#include "lsst/afw/table/aggregates.h"
#include "lsst/afw/table/IdFactory.h"
#include "lsst/afw/table/Catalog.h"
#include "lsst/afw/table/BaseColumnView.h"
#include "lsst/afw/table/io/FitsWriter.h"

namespace lsst { namespace afw {

namespace image {
class Wcs;
} // namespace image

namespace table {

/**
 *  @brief Bitflags to be passed to SourceCatalog::readFits and SourceCatalog::writeFits
 *
 *  Note that these flags may also be passed when reading/writing SourceCatalogs via the Butler,
 *  by passing a "flags" key/value pair as part of the data ID.
 */
enum SourceFitsFlags {
    SOURCE_IO_NO_FOOTPRINTS = 0x1,       ///< Do not read/write footprints at all
    SOURCE_IO_NO_HEAVY_FOOTPRINTS = 0x2  ///< Read/write heavy footprints as non-heavy footprints
};

typedef lsst::afw::detection::Footprint Footprint;

class SourceRecord;
class SourceTable;

/// @brief A collection of types that correspond to common measurements.
template <typename MeasTagT, typename ErrTagT>
struct Measurement {
    typedef MeasTagT MeasTag;  ///< the tag (template parameter) type used for the measurement
    typedef ErrTagT ErrTag;    ///< the tag (template parameter) type used for the uncertainty
    typedef typename Field<MeasTag>::Value MeasValue; ///< the value type used for the measurement
    typedef typename Field<ErrTag>::Value ErrValue;   ///< the value type used for the uncertainty
    typedef Key<MeasTag> MeasKey;  ///< the Key type for the actual measurement
    typedef Key<ErrTag> ErrKey;    ///< the Key type for the error on the measurement
    typedef FunctorKey<MeasTag> MeasFunctorKey;  ///< the Key type for the actual measurement
    typedef FunctorKey<ErrTag> ErrFunctorKey;    ///< the Key type for the error on the measurement
};

#ifndef SWIG

/// A collection of types useful for flux measurement algorithms.
struct Flux : public Measurement<double, double> {}; //pgee temporary

/// A collection of types useful for centroid measurement algorithms.
struct Centroid : public Measurement< Point<double>, Covariance< Point<float> > > {};

/// A collection of types useful for shape measurement algorithms.
struct Shape : public Measurement< Moments<double>, Covariance< Moments<float> > > {};

/// An enum for all the special flux aliases in Source.
enum FluxSlotEnum {
    FLUX_SLOT_PSF=0,
    FLUX_SLOT_MODEL,
    FLUX_SLOT_AP,
    FLUX_SLOT_INST,
    N_FLUX_SLOTS
};

struct FluxKeys {
    std::string name;
    Key<double> flux;
    Key<double> fluxSigma;
    Key<Flag> flag;
};
struct CentroidKeys {
    std::string name;
    lsst::afw::table::Point2DKey pos;
    lsst::afw::table::CovarianceMatrixKey<float,2> posErr;
    Key<Flag> flag;

    /// Default-constructor; all keys will be invalid.
    CentroidKeys() {}

    /// Main constructor.
    CentroidKeys(
    std::string const & name_,
    lsst::afw::table::Point2DKey const & pos_,
    lsst::afw::table::CovarianceMatrixKey<float,2> const & posErr_,
    Key<Flag> const & flag_
    ) : name(name_), pos(pos_), posErr(posErr_), flag(flag_) {}

    // No error constructor
    CentroidKeys(
    std::string const & name_,
    lsst::afw::table::Point2DKey const & pos_,
    Key<Flag> const & flag_
    ) : name(name_), pos(pos_), flag(flag_) {}
};
struct ShapeKeys {
    std::string name;
    lsst::afw::table::QuadrupoleKey quadrupole;
    lsst::afw::table::CovarianceMatrixKey<float,3> quadrupoleErr;
    Key<Flag> flag;
    
    /// Default-constructor; all keys will be invalid.
    ShapeKeys() {}

    /// Main constructor.
    ShapeKeys(
    std::string const & name_,
    lsst::afw::table::QuadrupoleKey const & quadrupole_,
    lsst::afw::table::CovarianceMatrixKey<float,3> const & quadrupoleErr_,
    Key<Flag> flag_
    ) : name(name_), quadrupole(quadrupole_), quadrupoleErr(quadrupoleErr_), flag(flag_) {}

    /// No error constructor.
    ShapeKeys(
    std::string const & name_,
    lsst::afw::table::QuadrupoleKey const & quadrupole_,
    Key<Flag> flag_
    ) : name(name_), quadrupole(quadrupole_), flag(flag_) {}
};
/**
 *  @brief A three-element tuple of measurement, uncertainty, and flag keys.
 *
 *  Most measurement should have more than one flag key to indicate different kinds of failures.
 *  This flag key should usually be set to be a logical OR of all of them, so it is set whenever
 *  a measurement cannot be fully trusted.
 */
template <typename MeasurementT>
struct KeyTuple {
    typename MeasurementT::MeasKey meas; ///< Key used for the measured value.
    typename MeasurementT::ErrKey err;   ///< Key used for the uncertainty.
    Key<Flag> flag;                      ///< Failure bit; set if the measurement did not fully succeed.

    /// Default-constructor; all keys will be invalid.
    KeyTuple() {}

    /// Main constructor.
    KeyTuple(
        typename MeasurementT::MeasKey const & meas_,
        typename MeasurementT::ErrKey const & err_,
        Key<Flag> const & flag_
    ) : meas(meas_), err(err_), flag(flag_) {}

};

/// Convenience function to setup fields for centroid measurement algorithms.
KeyTuple<Centroid> addCentroidFields(Schema & schema, std::string const & name, std::string const & doc);

/// Convenience function to setup fields for shape measurement algorithms.
KeyTuple<Shape> addShapeFields(Schema & schema, std::string const & name, std::string const & doc);

/// Convenience function to setup fields for flux measurement algorithms.
KeyTuple<Flux> addFluxFields(Schema & schema, std::string const & name, std::string const & doc);

#endif // !SWIG

template <typename RecordT> class SourceColumnViewT;

/**
 *  @brief Record class that contains measurements made on a single exposure.
 *
 *  Sources provide four additions to SimpleRecord / SimpleRecord:
 *   - Specific fields that must always be present, with specialized getters.
 *     The schema for a SourceTable should always be constructed by starting with the result of
 *     SourceTable::makeMinimalSchema.
 *   - A shared_ptr to a Footprint for each record.
 *   - A system of aliases (called slots) in which a SourceTable instance stores keys for particular
 *     measurements (a centroid, a shape, and a number of different fluxes) and SourceRecord uses
 *     this keys to provide custom getters and setters.  These are not separate fields, but rather
 *     aliases that can point to custom fields.
 */
class SourceRecord : public SimpleRecord {
public:

    typedef SourceTable Table;
    typedef SourceColumnViewT<SourceRecord> ColumnView;
    typedef SortedCatalogT<SourceRecord> Catalog;
    typedef SortedCatalogT<SourceRecord const> ConstCatalog;

    PTR(Footprint) getFootprint() const { return _footprint; }

    void setFootprint(PTR(Footprint) const & footprint) { _footprint = footprint; }

    CONST_PTR(SourceTable) getTable() const {
        return boost::static_pointer_cast<SourceTable const>(BaseRecord::getTable());
    }

    //@{
    /// @brief Convenience accessors for the keys in the minimal source schema.
    RecordId getParent() const;
    void setParent(RecordId id);
    //@}

    DECLARE_FLUX_GETTERS(`Psf')
    DECLARE_FLUX_GETTERS(`Model')
    DECLARE_FLUX_GETTERS(`Ap')
    DECLARE_FLUX_GETTERS(`Inst')
    DECLARE_CENTROID_GETTERS
    DECLARE_SHAPE_GETTERS

    /// @brief Return the centroid slot x coordinate.
    double getX() const;

    /// @brief Return the centroid slot y coordinate.
    double getY() const;

    /// @brief Return the shape slot Ixx value.
    double getIxx() const;

    /// @brief Return the shape slot Iyy value.
    double getIyy() const;

    /// @brief Return the shape slot Ixy value.
    double getIxy() const;

    /// @brief Update the coord field using the given Wcs and the field in the centroid slot.
    void updateCoord(image::Wcs const & wcs);

    /// @brief Update the coord field using the given Wcs and the image center from the given key.
    void updateCoord(image::Wcs const & wcs, Key< Point<double> > const & key);

protected:

    SourceRecord(PTR(SourceTable) const & table);

    virtual void _assign(BaseRecord const & other);

private:
    PTR(Footprint) _footprint;
};

/**
 *  @brief Table class that contains measurements made on a single exposure.
 *
 *  @copydetails SourceRecord
 */
class SourceTable : public SimpleTable {
public:

    typedef SourceRecord Record;
    typedef SourceColumnViewT<SourceRecord> ColumnView;
    typedef SortedCatalogT<Record> Catalog;
    typedef SortedCatalogT<Record const> ConstCatalog;

    /**
     *  @brief Construct a new table.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     *  @param[in] idFactory         Factory class to generate record IDs when they are not explicitly given.
     *                               If null, record IDs will default to zero.
     *
     *  Note that not passing an IdFactory at all will call the other override of make(), which will
     *  set the ID factory to IdFactory::makeSimple().
     */
    static PTR(SourceTable) make(Schema const & schema, PTR(IdFactory) const & idFactory);

    /**
     *  @brief Construct a new table.
     *
     *  @param[in] schema            Schema that defines the fields, offsets, and record size for the table.
     *
     *  This overload sets the ID factory to IdFactory::makeSimple().
     */
    static PTR(SourceTable) make(Schema const & schema) { return make(schema, IdFactory::makeSimple()); }

    /**
     *  @brief Return a minimal schema for Source tables and records.
     *
     *  The returned schema can and generally should be modified further,
     *  but many operations on sources will assume that at least the fields
     *  provided by this routine are present.
     *
     *  Keys for the standard fields added by this routine can be obtained
     *  from other static member functions of the SourceTable class.
     */
    static Schema makeMinimalSchema() { return getMinimalSchema().schema; }

    /**
     *  @brief Return true if the given schema is a valid SourceTable schema.
     *  
     *  This will always be true if the given schema was originally constructed
     *  using makeMinimalSchema(), and will rarely be true otherwise.
     */
    static bool checkSchema(Schema const & other) {
        return other.contains(getMinimalSchema().schema);
    }

    /// @brief Key for the parent ID.
    static Key<RecordId> getParentKey() { return getMinimalSchema().parent; }

    /// @copydoc BaseTable::clone
    PTR(SourceTable) clone() const { return boost::static_pointer_cast<SourceTable>(_clone()); }

    /// @copydoc BaseTable::makeRecord
    PTR(SourceRecord) makeRecord() { return boost::static_pointer_cast<SourceRecord>(_makeRecord()); }

    /// @copydoc BaseTable::copyRecord
    PTR(SourceRecord) copyRecord(BaseRecord const & other) {
        return boost::static_pointer_cast<SourceRecord>(BaseTable::copyRecord(other));
    }

    /// @copydoc BaseTable::copyRecord
    PTR(SourceRecord) copyRecord(BaseRecord const & other, SchemaMapper const & mapper) {
        return boost::static_pointer_cast<SourceRecord>(BaseTable::copyRecord(other, mapper));
    }

    DECLARE_FLUX_DEFINERS(`Psf')
    DECLARE_FLUX_DEFINERS(`Model')
    DECLARE_FLUX_DEFINERS(`Ap')
    DECLARE_FLUX_DEFINERS(`Inst')

    /**
     *  @brief Set the measurement used for the Centroid slot with a field name.
     *
     *  For version 0 tables, requires that the measurement adhere to the convention
     *  of having "<name>", "<name>.err", and "<name>.flags" fields for all three fields
     *  to be attached to slots.  Only the main measurement field is required.
     *  For version 1 tables: "<name>_x", "<name>_y", "<name>_xSigma", "<name>_ySigma"
     *  are the naming conventions
     */
    void defineCentroid(std::string const & name) {
        Schema schema = getSchema();
        _slotCentroid.name = name;
        if (getVersion() == 0) {
           Centroid::MeasKey measKey = schema[name];
           _slotCentroid.pos = lsst::afw::table::Point2DKey(measKey);
           try {
               Centroid::ErrKey errKey = schema[name]["err"];
               _slotCentroid.posErr = lsst::afw::table::CovarianceMatrixKey<float,2>(errKey);
           } catch (pex::exceptions::NotFoundError) {}
           try {
               _slotCentroid.flag = schema[name]["flags"];
           } catch (pex::exceptions::NotFoundError) {}
            return;
        }
        _slotCentroid.pos = lsst::afw::table::Point2DKey(schema[name+"_x"], schema[name+"_y"]);
        std::vector< Key<float> > sigma = std::vector< Key<float> >();
        std::vector< Key<float> > cov = std::vector< Key<float> >();
        try {
            _slotCentroid.flag = schema[name + "_flag"];
        } catch (pex::exceptions::NotFoundError) {}
        try {
            sigma.push_back(schema[name+"_xSigma"]);
            sigma.push_back(schema[name+"_ySigma"]);
            try {
                cov.push_back(schema[name+"_xyCov"]);
            } catch (pex::exceptions::NotFoundError) {}
            _slotCentroid.posErr = lsst::afw::table::CovarianceMatrixKey<float,2>(sigma, cov);
        } catch (pex::exceptions::NotFoundError) {}
/*  This code will go in place of the else clause immediately above when the SubSchema change is made
            _slotCentroid.name = name;
            _slotCentroid.pos = lsst::afw::table::Point2DKey(name);
            SubSchema sub = schema[name];
            try {
                CovarianceMatrixKey<float,2>::NameArray names = CovarianceMatrixKey<float,2>::NameArray(); 
                names.push_back("x");
                names.push_back("y");
                _slotCentroid.posErr = CovarianceMatrixKey<float,2>(sub, names);
            } catch (pex::exceptions::NotFoundError) {}
*/
    }
    
    /// @brief Return the name of the field used for the Centroid slot.
    std::string getCentroidDefinition() const {
        return _slotCentroid.name;
    }

    /// @brief Return the true if the Centroid slot is valid
    bool hasCentroidSlot() const {
        return _slotCentroid.pos.isValid();
    }
    /// @brief Return the key used for the Centroid slot.
    lsst::afw::table::Point2DKey getCentroidKey() const {
        return _slotCentroid.pos; 
    }

    /// @brief Return the key used for Centroid slot error or covariance.
    lsst::afw::table::CovarianceMatrixKey<float,2> getCentroidErrKey() const {
        return _slotCentroid.posErr; 
    }

    /// @brief Return the key used for the Centroid slot success flag.
    Key<Flag> getCentroidFlagKey() const {
            return _slotCentroid.flag;
    }

    /**
     *  @brief Set the measurement used for the Shape slot with a field name.
     *
     *  For version 0 tables, requires that the measurement adhere to the convention
     *  of having "<name>", "<name>.err", and "<name>.flags" fields for all three fields
     *  to be attached to slots.  Only the main measurement field is required.
     *  For version 1 tables: "<name>_xx", "<name>_yy", "<name>_xy"
     *                sigmas: "<name>_xxSigma", "<name>_yySigma", "<name>_xySigma"
     *                covariance: "<name>_xx_yy_Cov", "<name>_xx_xyCov", etc.
     */
    void defineShape(std::string const & name) {
        Schema schema = getSchema();
        _slotShape.name = name;
        if (getVersion() == 0) {
            Shape::MeasKey measKey = schema[name];
            _slotShape.quadrupole = lsst::afw::table::QuadrupoleKey(measKey.getIxx(),
                measKey.getIyy(), measKey.getIxy());
            try {
                Shape::ErrKey errKey = schema[name]["err"];
                _slotShape.quadrupoleErr = lsst::afw::table::CovarianceMatrixKey<float,3>(errKey);
            } catch (pex::exceptions::NotFoundError) {}
            try {
                _slotShape.flag = schema[name]["flags"];
            } catch (pex::exceptions::NotFoundError) {}
            return;
        }
        _slotShape.quadrupole = lsst::afw::table::QuadrupoleKey(
            schema[name + "_xx"],schema[name + "_yy"],schema[name + "_xy"]);
        try {
            _slotShape.flag = schema[name + "_flag"];
        } catch (pex::exceptions::NotFoundError) {}
        std::vector< Key<float> > sigma = std::vector< Key<float> >();
        std::vector< Key<float> > cov = std::vector< Key<float> >();
        try {
            sigma.push_back(schema[name+"_xxSigma"]);
            sigma.push_back(schema[name+"_yySigma"]);
            sigma.push_back(schema[name+"_xySigma"]);
            try {
                cov.push_back(schema[name + "_xx_yy_Cov"]);
                cov.push_back(schema[name + "_xx_xy_Cov"]);
                cov.push_back(schema[name + "_yy_xy_Cov"]);
            } catch (pex::exceptions::NotFoundError) {}
            _slotShape.quadrupoleErr = lsst::afw::table::CovarianceMatrixKey<float,3>(sigma, cov);
        } catch (pex::exceptions::NotFoundError) {}
    }
            
/*  This code will go in place of the else clause immediately above when the SubSchema change is made
            _slotShape.name = name;
            _slotShape.quadrupole = lsst::afw::table::QuadrupoleKey(name);
            SubSchema sub = schema[name];
            try {
                CovarianceMatrixKey<float,3>::NameArray names = CovarianceMatrixKey<float,3>::NameArray(); 
                names.push_back("xx");
                names.push_back("yy");
                names.push_back("xy");
                _slotShape.quadrupoleErr = CovarianceMatrixKey<float,3>(sub, names);
            } catch (pex::exceptions::NotFoundError) {}
*/
    /// @brief Return the name of the field used for the Shape slot.
    std::string getShapeDefinition() const {
        return _slotShape.name;
    }

    /// @brief Return the true if the Centroid slot is valid
    bool hasShapeSlot() const {
        return _slotShape.quadrupole.isValid();
    }

    /// @brief Return the key used for the Shape slot.
    lsst::afw::table::QuadrupoleKey getShapeKey() const {
        return _slotShape.quadrupole; 
    }

    /// @brief Return the key used for Shape slot error or covariance.
    lsst::afw::table::CovarianceMatrixKey<float,3> getShapeErrKey() const {
        return _slotShape.quadrupoleErr;
    }

    /// @brief Return the key used for the Shape slot success flag.
    Key<Flag> getShapeFlagKey() const {
        return _slotShape.flag;
    }

protected:

    SourceTable(Schema const & schema, PTR(IdFactory) const & idFactory);

    SourceTable(SourceTable const & other);

private:

    // Struct that holds the minimal schema and the special keys we've added to it.
    struct MinimalSchema {
        Schema schema;
        Key<RecordId> parent;

        MinimalSchema();
    };
    
    // Return the singleton minimal schema.
    static MinimalSchema & getMinimalSchema();

    friend class io::FitsWriter;

     // Return a writer object that knows how to save in FITS format.  See also FitsWriter.
    virtual PTR(io::FitsWriter) makeFitsWriter(fits::Fits * fitsfile, int flags) const;

    boost::array< FluxKeys, N_FLUX_SLOTS > _slotFlux; // aliases for flux measurements
    CentroidKeys _slotCentroid;  // alias for a centroid measurement
    ShapeKeys _slotShape;  // alias for a shape measurement
};

template <typename RecordT>
class SourceColumnViewT : public ColumnViewT<RecordT> {
public:

    typedef RecordT Record;
    typedef typename RecordT::Table Table;

    // See the documentation for BaseColumnView for an explanation of why these
    // accessors *appear* to violate const-correctness.

    DEFINE_FLUX_COLUMN_GETTERS(`Psf')
    DEFINE_FLUX_COLUMN_GETTERS(`Ap')
    DEFINE_FLUX_COLUMN_GETTERS(`Model')
    DEFINE_FLUX_COLUMN_GETTERS(`Inst')

    ndarray::Array<double,1> const getX() const {
        return this->operator[](this->getTable()->getCentroidKey().getX());
    }
    ndarray::Array<double,1> const getY() const {
        return this->operator[](this->getTable()->getCentroidKey().getY());
    }

    ndarray::Array<double,1> const getIxx() const {
        return this->operator[](this->getTable()->getShapeKey().getIxx());
    }
    ndarray::Array<double,1> const getIyy() const {
        return this->operator[](this->getTable()->getShapeKey().getIyy());
    }
    ndarray::Array<double,1> const getIxy() const {
        return this->operator[](this->getTable()->getShapeKey().getIxy());
    }

    /// @brief @copydoc BaseColumnView::make
    template <typename InputIterator>
    static SourceColumnViewT make(PTR(Table) const & table, InputIterator first, InputIterator last) {
        return SourceColumnViewT(BaseColumnView::make(table, first, last));
    }

protected:
    explicit SourceColumnViewT(BaseColumnView const & base) : ColumnViewT<RecordT>(base) {}
};

typedef SourceColumnViewT<SourceRecord> SourceColumnView;

#ifndef SWIG

DEFINE_FLUX_GETTERS(`Psf')
DEFINE_FLUX_GETTERS(`Model')
DEFINE_FLUX_GETTERS(`Ap')
DEFINE_FLUX_GETTERS(`Inst')

inline Centroid::MeasValue SourceRecord::getCentroid() const {
    return this->get(getTable()->getCentroidKey());
}

inline Centroid::ErrValue SourceRecord::getCentroidErr() const {
    return this->get(getTable()->getCentroidErrKey());
}

inline bool SourceRecord::getCentroidFlag() const {
    return this->get(getTable()->getCentroidFlagKey());
}

inline Shape::MeasValue SourceRecord::getShape() const {
    return this->get(getTable()->getShapeKey());
}

inline Shape::ErrValue SourceRecord::getShapeErr() const {
        return this->get(getTable()->getShapeErrKey());
}

inline bool SourceRecord::getShapeFlag() const {
    return this->get(getTable()->getShapeFlagKey());
}


inline RecordId SourceRecord::getParent() const { return get(SourceTable::getParentKey()); }
inline void SourceRecord::setParent(RecordId id) { set(SourceTable::getParentKey(), id); }
inline double SourceRecord::getX() const {
    return get(getTable()->getCentroidKey().getX());
}
inline double SourceRecord::getY() const {
    return get(getTable()->getCentroidKey().getY()); 
}
inline double SourceRecord::getIxx() const {
    return get(getTable()->getShapeKey().getIxx()); 
}
inline double SourceRecord::getIyy() const {
    return get(getTable()->getShapeKey().getIyy()); 
}
inline double SourceRecord::getIxy() const {
    return get(getTable()->getShapeKey().getIxy()); 
}

#endif // !SWIG

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Source_h_INCLUDED
