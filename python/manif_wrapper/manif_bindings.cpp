#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>

#include "manif/SE3.h"


namespace py = pybind11;

#define WRAP_LIE_GROUP(T) \
  .def(py::init<>()) \
  .def(py::init<const T::Transformation&>()) \
  .def("log", (T::Tangent (T::*)() const) &T::log) \
  .def_static("exp", &T::exp) \
  .def_static("vee", &T::vee) \
  .def_static("hat", &T::hat) \
  .def("inverse", &T::inverse) \
  .def("params", &T::params) \
  .def("matrix", &T::matrix) \
  .def("Adj", &T::Adj) \
  .def(py::self * py::self) \
  .def(py::self * T::Point()) \
  .def("__imul__", [](const T& a, const T& b) { \
        return T(a * b); \
      }, py::is_operator())


template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

template <typename LieGroupType>
void wrap_lie_group(py::class_<LieGroupType>& py_class){
    py_class.def(py::init<>());
    py_class.def("setIdentity",&LieGroupType::setIdentity);
    py_class.def("setRandom",&LieGroupType::setRandom);
  py_class.def("inverse",&LieGroupType::inverse);
  py_class.def("log",&LieGroupType::log);
//  py_class.def("compose",&LieGroupType::compose<LieGroupType>);
  //py_class.def("act",static_cast<manif::SE3d::Vector (manif::SE3d::*)(const manif::SE3d::Vector&,manif::SE3d::OptJacobianRef,manif::SE3d::OptJacobianRef)>(&LieGroupType::act));
  //py_class.def("act",&LieGroupType::act);
  py_class.def("adj",&LieGroupType::adj);
  py_class.def("compose",&LieGroupType::template compose<LieGroupType>);
}

template <typename LieGroupType,typename TangentType>
void wrap_combo_funcs(py::class_<LieGroupType>& base,py::class_<TangentType>& tanget){

  base.def("rplus",&LieGroupType::template rplus<TangentType>);
  base.def("lplus",&LieGroupType::template lplus<TangentType>);
  base.def("plus",&LieGroupType::template plus<TangentType>);
  base.def("rminus",&LieGroupType::template rminus<LieGroupType>);
  base.def("lminus",&LieGroupType::template lminus<LieGroupType>);
  base.def("minus",&LieGroupType::template minus<LieGroupType>);
  base.def("between",&LieGroupType::template between<LieGroupType>);
  base.def("isApprox",&LieGroupType::template isApprox<LieGroupType>);


}

template <typename TangentType>
void wrap_tanget(py::class_<TangentType>& py_class){
  py_class.def(py::init<>());
  py_class.def("setZero",&TangentType::setZero);
  py_class.def("setRandom",&TangentType::setRandom);
  //py_class.def("w",&TangentType::w);
  py_class.def("weightedNorm",&TangentType::weightedNorm);
  py_class.def("hat",&TangentType::hat);
  py_class.def("exp",&TangentType::exp);
  py_class.def("rplus",&TangentType::rplus);
  py_class.def("lplus",&TangentType::lplus);
  //py_class.def("plus",&TangentType::plus);
  py_class.def("rjac",&TangentType::rjac);
  py_class.def("ljac",&TangentType::ljac);
  py_class.def("smallAdj",&TangentType::smallAdj);
}


PYBIND11_MODULE(PyManif, m) {
  m.doc() = "Manif wrappers"; // optional module docstring'

  py::class_<manif::SE3d> se3d(m, "SE3d");

  se3d.def("transform",&manif::SE3d::transform);
  se3d.def("isometry",&manif::SE3d::isometry);
  se3d.def("quat",static_cast<manif::SE3d::QuaternionDataType (manif::SE3d::*)() const>(&manif::SE3d::quat));
  se3d.def("translation",static_cast<manif::SE3d::Translation (manif::SE3d::*)() const>(&manif::SE3d::translation));
  se3d.def("x",&manif::SE3d::x);
  se3d.def("y",&manif::SE3d::y);
  se3d.def("z",&manif::SE3d::z);
  se3d.def("normalize",&manif::SE3d::normalize);
  se3d.def("transform",&manif::SE3d::transform);
  se3d.def("quat",static_cast<void (manif::SE3d::*)(const manif::SE3d::QuaternionDataType&)>(&manif::SE3d::quat));
  se3d.def("quat",static_cast<void (manif::SE3d::*)(const manif::SO3d&)>(&manif::SE3d::quat));
  se3d.def("quat",static_cast<void (manif::SE3d::*)(const manif::SE3d::Translation&)>(&manif::SE3d::translation));

  wrap_lie_group<manif::SE3d>(se3d);


  py::class_<manif::SE3Tangentd> se3d_tanget(m, "SE3Tangentd");

  wrap_tanget<manif::SE3Tangentd>(se3d_tanget);

  wrap_combo_funcs<manif::SE3d,manif::SE3Tangentd>(se3d,se3d_tanget);
  
//  se3d.def("act",static_cast<manif::SE3d::Vector (manif::SE3d::*)(const manif::SE3d::Vector& v,
//                                                                  manif::SE3d::OptJacobianRef J_vout_m,
//                                                                  manif::SE3d::OptJacobianRef J_vout_v)>(&manif::SE3d::act));

}
