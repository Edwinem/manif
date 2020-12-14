#pragma once


#include "lt/optional.hpp"
#include <pybind11/stl.h>

namespace pybind11 { namespace detail {

// This type caster is intended to be used for std::optional and std::experimental::optional
template<typename T> struct optional_caster_custom {
  using value_conv = make_caster<typename T::value_type>;

  template<typename T_>
  static handle cast(T_ &&src, return_value_policy policy, handle parent) {
    if (!src)
      return none().inc_ref();
    if (!std::is_lvalue_reference<T>::value) {
      policy = return_value_policy_override<T>::policy(policy);
    }
    return value_conv::cast(*std::forward<T_>(src), policy, parent);
  }

  bool load(handle src, bool convert) {
    if (!src) {
      return false;
    } else if (src.is_none()) {
      return true;  // default-constructed value is already empty
    }
    value_conv inner_caster;
    if (!inner_caster.load(src, convert))
      return false;

    value.emplace(cast_op<typename T::value_type &&>(std::move(inner_caster)));
    return true;
  }

PYBIND11_TYPE_CASTER(T, _("TLOptional"));
};

template<typename T> struct type_caster<tl::optional<T>>
    : public optional_caster_custom<tl::optional<T>> {
};

template<> struct type_caster<tl::monostate>
    : public void_caster<tl::nullopt_t> {
};

}}