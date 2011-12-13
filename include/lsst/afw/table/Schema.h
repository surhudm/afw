// -*- lsst-c++ -*-
#ifndef AFW_TABLE_Schema_h_INCLUDED
#define AFW_TABLE_Schema_h_INCLUDED

#include <set>

#include "boost/shared_ptr.hpp"
#include "boost/ref.hpp"

#include "lsst/ndarray.h"
#include "lsst/afw/table/Key.h"
#include "lsst/afw/table/Field.h"
#include "lsst/afw/table/detail/SchemaData.h"
#include "lsst/afw/table/Flag.h"

namespace lsst { namespace afw { namespace table {

class SchemaProxy;

/**
 *  @brief Defines the fields and offsets for a table.
 *
 *  Schema behaves like a container of SchemaItem objects, mapping a descriptive Field object
 *  with the Key object used to access record and ColumnView values.  A Schema is the most
 *  important ingredient in creating a table.
 *
 *  Because offsets for fields are assigned when the field is added to the Schema, 
 *  Schemas do not support removing fields.
 *
 *  A SchemaMapper object can be used to define a relationship between two Schemas to be used
 *  when copying values from one table to another or loading/saving selected fields to disk.
 *
 *  Schema uses copy-on-write, and hence should always be held by value rather than smart pointer.
 *  When creating a Python interface, functions that return Schema by const reference should be
 *  converted to return by value to ensure proper memory management and encapsulation.
 */
class Schema {
    typedef detail::SchemaData Data;
public:

    /// @brief Set type returned by describe().
    typedef std::set<FieldDescription> Description;

    /// @brief Return true if the schema constains space for a parent ID field.
    bool hasTree() const { return _data->_hasTree; }

    /// @brief Find a SchemaItem in the Schema by name.
    template <typename T>
    SchemaItem<T> find(std::string const & name) const {
        return _data->find<T>(name);
    }

    /// @brief Find a SchemaItem in the Schema by key.
    template <typename T>
    SchemaItem<T> find(Key<T> const & key) const;

    /**
     *  @brief Lookup a (possibly incomplete) name in the Schema.
     *
     *  See SchemaProxy for more information.
     *
     *  This member function should generally only be used on
     *  "finished" Schemas; modifying a Schema after a SchemaProxy
     *  to it has been constructed will not allow the proxy to track
     *  the additions, and will invoke the copy-on-write
     *  mechanism of the Schema itself.
     */
    SchemaProxy operator[](std::string const & name) const;

    /**
     *  @brief Return a vector of field names in the schema.
     *
     *  If topOnly==true, return a unique list of only the part
     *  of the names before the first period.  For example,
     *  if the full list of field names is ['a.b.c', 'a.d', 'e.f'],
     *  topOnly==true will return ['a', 'e'].
     */
    std::vector<std::string> getNames(bool topOnly=false) const;

    /**
     *  @brief Return a set with descriptions of all the fields.
     *
     *  The set will be ordered by field name, not by Key.
     */
    Description describe() const;

    /// @brief Return the raw size of a record in bytes.
    int getRecordSize() const { return _data->_recordSize; }

    /**
     *  @brief Add a new field to the Schema, and return the associated Key.
     *
     *  The offsets of fields are determined by the order they are added, but
     *  may be not contiguous (the Schema may add padding to align fields, and how
     *  much padding is considered an implementation detail).
     */
    template <typename T>
    Key<T> addField(Field<T> const & field);    

    /**
     *  @brief Add a new field to the Schema, and return the associated Key.
     *
     *  This is simply a convenience wrapper, equivalent to:
     *  @code
     *  addField(Field<T>(name, doc, units, base))
     *  @endcode
     */
    template <typename T>
    Key<T> addField(
        std::string const & name, std::string const & doc, std::string const & units = "",
        FieldBase<T> const & base = FieldBase<T>()
    ) {
        return addField(Field<T>(name, doc, units, base));
    }

    /**
     *  @brief Add a new field to the Schema, and return the associated Key.
     *
     *  This is simply a convenience wrapper, equivalent to:
     *  @code
     *  addField(Field<T>(name, doc, base))
     *  @endcode
     */
    template <typename T>
    Key<T> addField(std::string const & name, std::string const & doc, FieldBase<T> const & base) {
        return addField(Field<T>(name, doc, base));
    }

    /// @brief Replace the Field (name/description) for an existing Key.
    template <typename T>
    void replaceField(Key<T> const & key, Field<T> const & field);

    /**
     *  @brief Apply a functor to each SchemaItem in the Schema.
     *
     *  The functor must have a templated or sufficiently overloaded operator() that supports
     *  SchemaItems of all supported field types - even those that are not present in this
     *  particular Schema.
     *
     *  The functor will be passed by value by default; use boost::ref to pass it by reference.
     */
    template <typename F>
    void forEach(F func) const {
        Data::VisitorWrapper<typename boost::unwrap_reference<F>::type &> visitor(func);
        std::for_each(_data->_items.begin(), _data->_items.end(), visitor);
    }

    /// @brief Construct an empty Schema.
    explicit Schema(bool hasTree);

private:

    friend class detail::Access;
    friend class SchemaProxy;
    
    /// @brief Copy on write; should be called by all mutators.
    void _edit();

    template <typename T>
    Key<T> _addField(Field<T> const & field);    

    Key<Flag> _addField(Field<Flag> const & field);

    boost::shared_ptr<Data> _data;
};

/**
 *  @brief A proxy type for name lookups in a Schema.
 *
 *  Elements of schema names are assumed to be separated by
 *  periods ("a.b.c.d"); an incomplete lookup is one that
 *  does not resolve to a field.  Not that even complete
 *  lookups can have nested names; a Point field, for instance,
 *  has "x" and "y" nested names.
 *
 *  This proxy object is implicitly convertible to both
 *  the appropriate Key type and the appropriate Field type,
 *  if the name is a complete one, and supports additional
 *  find() operations for nested names.  It also provides
 *  accessors that forward to the getters of Field.
 *
 *  SchemaProxy is implemented as a proxy that essentially
 *  calls Schema::find after concatenating strings.  It
 *  does not provide any performance advantage over using
 *  Schema::find directly.  It is also lazy, so looking up
 *  a name prefix that does not exist within the schema
 *  is not considered an error until the proxy is used.
 *
 *  Some examples:
 *  @code
 *  Schema schema(false);
 *  Key<int> a_i = schema.addField<int>("a.i", "integer field");
 *  Key< Point<double> > a_p = schema.addField< Point<double> >("a.p", "point field");
 *  
 *  assert(schema["a.i"] == a_i);
 *  SchemaProxy a = schema["a"];
 *  assert(a["i"] == a_i);
 *  Field<int> f_a_i = schema["a.i"];
 *  assert(f_a_i.getDoc() == "integer field");
 *  assert(schema["a.i"].getName() == "a.i");
 *  assert(schema.find("a.p.x") == a_p.getX());
 *  @endcode
 */
class SchemaProxy {
    typedef detail::SchemaData Data;
public:
    
    /// @brief Find a nested SchemaItem by name.
    template <typename T>
    SchemaItem<T> find(std::string const & name) const;

    /// @brief Return a nested proxy.
    SchemaProxy operator[](std::string const & name) const;

    /**
     *  @brief Return a vector of nested names that start with the SchemaProxy's prefix.
     *
     *  @sa Schema::getNames
     */
    std::vector<std::string> getNames(bool topOnly=false) const;

    template <typename T>
    operator Key<T>() const { return _data->find<T>(_name).key; }

    template <typename T>
    operator Field<T>() const { return _data->find<T>(_name).field; }

    std::string const & getName() const { return _name; }

    std::string const & getDoc() const;

    std::string const & getUnits() const;

private:

    friend class Schema;

    SchemaProxy(PTR(Data) const & data, std::string const & name) :
        _data(data), _name(name)
    {}

    boost::shared_ptr<Data> _data;
    std::string _name;
};

}}} // namespace lsst::afw::table

#endif // !AFW_TABLE_Schema_h_INCLUDED
