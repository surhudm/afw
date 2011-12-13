// -*- lsst-c++ -*-
#ifndef AFW_TABLE_RecordInterface_h_INCLUDED
#define AFW_TABLE_RecordInterface_h_INCLUDED

#include "boost/iterator/transform_iterator.hpp"

#include "lsst/base.h"
#include "lsst/afw/table/RecordBase.h"
#include "lsst/afw/table/IteratorBase.h"
#include "lsst/afw/table/detail/Access.h"

namespace lsst { namespace afw { namespace table {

/**
 *  @brief A facade base class that provides most of the public interface of a record.
 *
 *  RecordInterface inherits from RecordBase and gives a record a consistent public interface
 *  by providing thin wrappers around RecordBase member functions that return adapted types.
 *
 *  Final record classes should inherit from RecordInterface, templated on a tag class that
 *  typedefs both the record class and the corresponding final record class (see SimpleRecord
 *  for an example).
 *
 *  RecordInterface does not provide public wrappers for member functions that add new records,
 *  because final record classes may want to control what auxiliary data is required to be present
 *  in a record.
 */
template <typename Tag>
class RecordInterface : public RecordBase {
public:

    typedef typename Tag::Record Record;
    typedef boost::transform_iterator<detail::RecordConverter<Record>,IteratorBase> Iterator;

    /// @copydoc RecordBase::_getParent
    Record getParent() const {
        return detail::Access::makeRecord<Record>(this->_getParent());
    }

protected:

    template <typename OtherTag> friend class TableInterface;

    explicit RecordInterface(RecordBase const & other) : RecordBase(other) {}

    RecordInterface(RecordInterface const & other) : RecordBase(other) {}

    void operator=(RecordInterface const & other) { RecordBase::operator=(other); }

};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_RecordInterface_h_INCLUDED
