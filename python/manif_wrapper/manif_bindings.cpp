#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>

#include "manif/SE3.h"

namespace py = pybind11;


template<typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;



//template<typename TangentType,
//    typename std::enable_if<std::is_integral<TangentType>::value, int>::type = 0>
//void add_rjacinv(py::class_<TangentType> &tanget)
//-> void
//{
//  std::cout << "I'm an integer!\n";
//}
//
//template<typename T,
//    typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
//auto foo(T)
//-> void
//{
//  std::cout << "I'm a floating point number!\n";
//}





template<typename LieGroupType>
void wrap_lie_group(py::class_<LieGroupType> &py_class) {
  py_class.def(py::init<>());
  py_class.def("setIdentity", &LieGroupType::setIdentity);
  py_class.def("setRandom", &LieGroupType::setRandom);
  py_class.def("inverse", &LieGroupType::inverse);
  py_class.def("log", &LieGroupType::log);
//  py_class.def("compose",&LieGroupType::compose<LieGroupType>);
  //py_class.def("act",static_cast<manif::SE3d::Vector (manif::SE3d::*)(const manif::SE3d::Vector&,manif::SE3d::OptJacobianRef,manif::SE3d::OptJacobianRef)>(&LieGroupType::act));
  //py_class.def("act",&LieGroupType::act);
  py_class.def("adj", &LieGroupType::adj);
  py_class.def("compose", &LieGroupType::template compose<LieGroupType>);
}

template<typename LieGroupType, typename TangentType>
void wrap_combo_funcs(py::class_<LieGroupType> &base, py::class_<TangentType> &tanget) {

  base.def("rplus", &LieGroupType::template rplus<TangentType>);
  base.def("lplus", &LieGroupType::template lplus<TangentType>);
  base.def("plus", &LieGroupType::template plus<TangentType>);
  base.def("rminus", &LieGroupType::template rminus<LieGroupType>);
  base.def("lminus", &LieGroupType::template lminus<LieGroupType>);
  base.def("minus", &LieGroupType::template minus<LieGroupType>);
  base.def("between", &LieGroupType::template between<LieGroupType>);
  base.def("isApprox", &LieGroupType::template isApprox<LieGroupType>);

  // LieGroupBase operator overloads
  base.def("__add__", [](const LieGroupType &a, const TangentType& b) {
    return a + b;
  }, py::is_operator());
  base.def("__iadd__", [](LieGroupType &a, const TangentType& b) {
    a += b;
  }, py::is_operator());
  base.def("__sub__", [](const LieGroupType &a, const LieGroupType& b) {
    return a - b;
  }, py::is_operator());
  base.def("__mul__", [](const LieGroupType &a, const LieGroupType& b) {
    return a * b;
  }, py::is_operator());
  base.def("__imul__", [](LieGroupType &a, const LieGroupType& b) {
    a *= b;
  }, py::is_operator());


  // Tangent group operator overloads
  tanget.def("__neg__", [](TangentType &a) {
    return -a;
  }, py::is_operator());
  tanget.def("__add__", [](const TangentType &a, const LieGroupType& b) {
    return a + b;
  }, py::is_operator());
  tanget.def("__iadd__", [](TangentType &a, const TangentType& b) {
    a += b;
  }, py::is_operator());
  tanget.def("__isub__", [](TangentType &a, const TangentType& b) {
    return a -= b;
  }, py::is_operator());
  tanget.def("__iadd__", [](TangentType &a, const typename manif::internal::traits<TangentType>::DataType& b) {
    a += b;
  }, py::is_operator());
  tanget.def("__isub__", [](TangentType &a, const typename manif::internal::traits<TangentType>::DataType& b) {
    a -= b;
  }, py::is_operator());
  tanget.def("__imul__", [](TangentType &a, const typename TangentType::Scalar b) {
    return a *= b;
  }, py::is_operator());
  tanget.def("__itruediv__", [](TangentType &a, const typename TangentType::Scalar b) {
    return a /= b;
  }, py::is_operator());

}


template<typename TangentType>
void wrap_tanget(py::class_<TangentType> &py_class) {
  py_class.def(py::init<>());
  py_class.def("from_arr",[](TangentType& myself,const typename manif::internal::traits<TangentType>::DataType& vector){
    myself = vector;
  });
  py_class.def("setZero", &TangentType::setZero);
  py_class.def("setRandom", &TangentType::setRandom);
//  py_class.def("w",&TangentType::w);
  py_class.def("weightedNorm", &TangentType::weightedNorm);
  py_class.def("hat", &TangentType::hat);
  py_class.def("exp", &TangentType::exp);
  py_class.def("rplus", &TangentType::rplus);
  py_class.def("lplus", &TangentType::lplus);
  py_class.def("plus",
               static_cast<typename TangentType::LieGroup (TangentType::*)(const typename TangentType::LieGroup &,
                                                                           typename TangentType::OptJacobianRef,
                                                                           typename TangentType::OptJacobianRef) const>(&TangentType::plus));
  py_class.def("plus", (&TangentType::template plus<TangentType>));
  py_class.def("minus", (&TangentType::template plus<TangentType>));
  py_class.def("rjac", &TangentType::rjac);
  py_class.def("ljac", &TangentType::ljac);
  py_class.def("smallAdj", &TangentType::smallAdj);


  // Can't figure out how to do static_cast overload resolution and specialize with the templates. So we just set it up
  // as a lambda function, and do it manually.
  py_class.def("isApprox",[](const TangentType& myself,const TangentType& other,typename TangentType::Scalar eps){
    return myself.isApprox(other,eps);
  });
  py_class.def("isApprox",[](const TangentType& myself,const Eigen::MatrixXd& other,typename TangentType::Scalar eps){
    return myself.isApprox(other,eps);
  });

}


template <typename TangentType>
TangentType create_tangent_from_vector(const typename manif::internal::traits<TangentType>::DataType& vec) {
  TangentType tangent=vec;
  return tangent;
}


PYBIND11_MODULE(PyManif, m) {
  m.doc() = "Manif wrappers"; // optional module docstring'

  py::class_<manif::SE3d> se3d(m, "SE3d");


  se3d.def(py::init<const manif::SE3d::Translation&,  const Eigen::Quaternion<manif::SE3d::Scalar>&>());
  se3d.def(py::init<const manif::SE3d::Translation&,  const Eigen::AngleAxis<manif::SE3d::Scalar>&>());
  se3d.def(py::init<const manif::SE3d::Translation&,  const manif::SO3d&>());
  se3d.def(py::init<const double, const double, const double,                          const double, const double, const double>());
  se3d.def(py::init<const Eigen::Transform<manif::SE3d::Scalar,3,Eigen::Isometry>>());

  se3d.def("transform", &manif::SE3d::transform);
  se3d.def("isometry", &manif::SE3d::isometry);
  se3d.def("quat", static_cast<manif::SE3d::QuaternionDataType (manif::SE3d::*)() const>(&manif::SE3d::quat));
  se3d.def("translation", static_cast<manif::SE3d::Translation (manif::SE3d::*)() const>(&manif::SE3d::translation));
  se3d.def("x", &manif::SE3d::x);
  se3d.def("y", &manif::SE3d::y);
  se3d.def("z", &manif::SE3d::z);
  se3d.def("normalize", &manif::SE3d::normalize);
  se3d.def("transform", &manif::SE3d::transform);
  se3d.def("quat", static_cast
      <void (manif::SE3d::*)(const manif::SE3d::QuaternionDataType &)>(&manif::SE3d::quat)
  );
  se3d.def("quat", static_cast
      <void (manif::SE3d::*)(const manif::SO3d &)>(&manif::SE3d::quat)
  );
  se3d.def("quat", static_cast
      <void (manif::SE3d::*)(const manif::SE3d::Translation &)>(&manif::SE3d::translation)
  );

  se3d.def("__str__", [](const manif::SE3d& my_self) {
    std::stringstream ss;
    ss << my_self.rotation();
    std::string rot_str = ss.str();
    py::str trans = py::str("{},{},{}").format(my_self.x(), my_self.y(), my_self.z());
    auto quat = my_self.quat();
    py::str quat_str = py::str("w:{},x:{},y:{},z:{}").format(quat.w(), quat.x(), quat.y(), quat.z());
    Eigen::Vector3d euler = quat.toRotationMatrix().eulerAngles(2, 1, 0);
    py::str euler_str = py::str("roll:{},pitch:{},yaw:{}").format(euler[2], euler[1], euler[0]);

    return py::str("Rotation Matrix:\n{}\nQuaternion:\n{}\nRPY:\n{}\nTrans:[{}]").format(rot_str, quat_str, euler_str, trans);

  });


  wrap_lie_group<manif::SE3d>(se3d);

  py::class_<manif::SE3Tangentd> se3d_tanget(m, "SE3Tangentd");
  se3d_tanget.def(py::init<>(&create_tangent_from_vector<manif::SE3Tangentd>));
  se3d_tanget.def("__str__", [](const manif::SE3Tangentd& my_self) {
    return py::str("{}({},{},{},{},{},{})").format(
        "SE3Tangentd",
        my_self.coeffs()(0),
        my_self.coeffs()(1),
        my_self.coeffs()(2),
    my_self.coeffs()(3),
    my_self.coeffs()(4),
        my_self.coeffs()(5));
  });

  wrap_tanget<manif::SE3Tangentd>(se3d_tanget);
//se3d_tanget.def("plus",static_cast<manif::SE3Tangentd::LieGroup (manif::SE3Tangentd::*)(const manif::SE3Tangentd::LieGroup&,manif::SE3Tangentd::OptJacobianRef,
//                                                                                  manif::SE3Tangentd::OptJacobianRef) const>(&manif::SE3Tangentd::plus));

  wrap_combo_funcs<manif::SE3d, manif::SE3Tangentd>(se3d, se3d_tanget);


//  se3d.def("act",static_cast<manif::SE3d::Vector (manif::SE3d::*)(const manif::SE3d::Vector& v,
//                                                                  manif::SE3d::OptJacobianRef J_vout_m,
//                                                                  manif::SE3d::OptJacobianRef J_vout_v)>(&manif::SE3d::act));

  using T=double ;
  using Class = Eigen::Quaternion<T>;
  py::class_<Class> py_class(m, "Quaternion");
  py_class.attr("__doc__") =
      "Provides a unit quaternion binding of Eigen::Quaternion<>.";
  py::object py_class_obj = py_class;
  py_class
      .def(py::init([]() {
        return Class::Identity();
      }))
      .def_static("Identity", []() {
        return Class::Identity();
      })
      .def(py::init([](const Eigen::Vector4d& wxyz) {
        Class out(wxyz(0), wxyz(1), wxyz(2), wxyz(3));
        return out;
      }), py::arg("wxyz"))
      .def(py::init([](T w, T x, T y, T z) {
        Class out(w, x, y, z);
        return out;
      }), py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"))
      .def(py::init([](const Eigen::Matrix3d& rotation) {
        Class out(rotation);
        return out;
      }), py::arg("rotation"))
      .def(py::init([](const Class& other) {
        return other;
      }), py::arg("other"))

      .def("w", [](const Class* self) { return self->w(); })
      .def("x", [](const Class* self) { return self->x(); })
      .def("y", [](const Class* self) { return self->y(); })
      .def("z", [](const Class* self) { return self->z(); })
      .def("xyz", [](const Class* self) { return self->vec(); })
      .def("wxyz", [](Class* self) {
        Eigen::Vector4d wxyz;
        wxyz << self->w(), self->vec();
        return wxyz;
      })
      .def("set_wxyz", [](Class* self, const Eigen::Vector4d& wxyz) {
        Class update;
        update.w() = wxyz(0);
        update.vec() = wxyz.tail(3);
        *self = update;
      }, py::arg("wxyz"))
      .def("set_wxyz", [](Class* self, T w, T x, T y, T z) {
        Class update(w, x, y, z);
        *self = update;
      }, py::arg("w"), py::arg("x"), py::arg("y"), py::arg("z"))
      .def("rotation", [](const Class* self) {
        return self->toRotationMatrix();
      })
      .def("set_rotation", [](Class* self, const Eigen::Matrix3d& rotation) {
        Class update(rotation);
        *self = update;
      })
      .def("slerp",[](Class& q, double & t, Class other) {
        return q.slerp(t, other);
      })

      .def("__str__", [py_class_obj](const Class* self) {
        return py::str("{}(w={}, x={}, y={}, z={})").format(
            py_class_obj.attr("__name__"),
            self->w(), self->x(), self->y(), self->z());
      })

      .def("__repr__", [](const Eigen::Quaterniond &v) {
        std::ostringstream oss;
        oss << "(" << v.w() << ", " << v.x() << ", " << v.y() << ", " << v.z() << ")";
        return oss.str();
      })
          // Do not define operator `__mul__` until we have the Python3 `@`
          // operator so that operations are similar to those of arrays.
      .def("multiply", [](const Class& self, const Class& other) {
        return self * other;
      })
      .def("multiply", [](const Class& self, const Eigen::Vector3d& position) {
        return self * position;
      }, py::arg("position"))
      .def("inverse", [](const Class* self) {
        return self->inverse();
      })
      .def("conjugate", [](const Class* self) {
        return self->conjugate();
      })
      .def("__mul__", [](const Class& self, const Class& other) {
        return self * other;
      },py::is_operator())
      .def("__mul__", [](const Class& self, const Eigen::Vector3d& position) {
        return self * position;
      }, py::arg("position"),py::is_operator())


      ;


}

