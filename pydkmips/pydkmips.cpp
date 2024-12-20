#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstring>
#include <sys/types.h>
#include <omp.h>

#include "greedy.h"
#include "dual_greedy.h"
#include "bc_greedy.h"
#include "bc_dual_greedy.h"

namespace py = pybind11;
using namespace ip;

PYBIND11_MODULE(_pydkmips_impl, m) {
    m.doc() = "C++ implementation module for DkMIPS";

    py::class_<Greedy>(m, "Greedy")
        .def(py::init([](int n, int d, int d2, py::array_t<float> items, py::array_t<float> i2v) {
            py::buffer_info buf = items.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Number of dimensions must be 2");
            if (buf.shape[0] != n || buf.shape[1] != d)
                throw std::runtime_error("Input shape must match n and d");
            py::buffer_info buf2 = i2v.request();
            if (buf2.ndim != 2)
                throw std::runtime_error("Number of dimensions must be 2");
            if (buf2.shape[0] != n || buf2.shape[1] != d2)
                throw std::runtime_error("Input shape must match n and d2");
            return new Greedy(n, d, d2, static_cast<float*>(buf.ptr), static_cast<float*>(buf2.ptr));
        }))
        .def("dkmips_avg", [](Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_avg(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_plus_avg", [](Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_plus_avg(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_max", [](Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_max(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_plus_max", [](Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_plus_max(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_avg_i2v", [](Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_avg_i2v(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_plus_avg_i2v", [](Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_plus_avg_i2v(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_max_i2v", [](Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_max_i2v(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_plus_max_i2v", [](Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_plus_max_i2v(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        });
    
    py::class_<Dual_Greedy>(m, "Dual_Greedy")
        .def(py::init([](int n, int d, int d2, py::array_t<float> items, py::array_t<float> i2v) {
            py::buffer_info buf = items.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Number of dimensions must be 2");
            if (buf.shape[0] != n || buf.shape[1] != d)
                throw std::runtime_error("Input shape must match n and d");
            py::buffer_info buf2 = i2v.request();
            if (buf2.ndim != 2)
                throw std::runtime_error("Number of dimensions must be 2");
            if (buf2.shape[0] != n || buf2.shape[1] != d2)
                throw std::runtime_error("Input shape must match n and d2");
            return new Dual_Greedy(n, d, d2, static_cast<float*>(buf.ptr), static_cast<float*>(buf2.ptr));
        }))
        .def("dkmips_avg", [](Dual_Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_avg(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_plus_avg", [](Dual_Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_plus_avg(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_max", [](Dual_Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_max(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_plus_max", [](Dual_Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_plus_max(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_avg_i2v", [](Dual_Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_avg_i2v(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_plus_avg_i2v", [](Dual_Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_plus_avg_i2v(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_max_i2v", [](Dual_Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_max_i2v(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_plus_max_i2v", [](Dual_Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_plus_max_i2v(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        });

    py::class_<BC_Greedy>(m, "BC_Greedy")
        .def(py::init([](int n, int d, int d2, py::array_t<float> items, py::array_t<float> i2v) {
            py::buffer_info buf = items.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Number of dimensions must be 2");
            if (buf.shape[0] != n || buf.shape[1] != d)
                throw std::runtime_error("Input shape must match n and d");
            py::buffer_info buf2 = i2v.request();
            if (buf2.ndim != 2)
                throw std::runtime_error("Number of dimensions must be 2");
            if (buf2.shape[0] != n || buf2.shape[1] != d2)
                throw std::runtime_error("Input shape must match n and d2");
            return new BC_Greedy(n, d, d2, static_cast<float*>(buf.ptr), static_cast<float*>(buf2.ptr));
        }))
        .def("dkmips_avg", [](BC_Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_avg(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_max", [](BC_Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_max(k, lambda, c, query_ptr);
            
            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }

            delete[] result;
            return pyResult;
        })
        .def("dkmips_avg_i2v", [](BC_Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_avg_i2v(k, lambda, c, query_ptr);

            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }
            
            delete[] result;
            return pyResult;
        })
        .def("dkmips_max_i2v", [](BC_Greedy& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_max_i2v(k, lambda, c, query_ptr);

            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }

            delete[] result;
            return pyResult;
        });

    py::class_<BC_Dual>(m, "BC_Dual")
        .def(py::init([](int n, int d, int d2, py::array_t<float> items, py::array_t<float> i2v) {
            py::buffer_info buf = items.request();
            if (buf.ndim != 2)
                throw std::runtime_error("Number of dimensions must be 2");
            if (buf.shape[0] != n || buf.shape[1] != d)
                throw std::runtime_error("Input shape must match n and d");
            py::buffer_info buf2 = i2v.request();
            if (buf2.ndim != 2)
                throw std::runtime_error("Number of dimensions must be 2");
            if (buf2.shape[0] != n || buf2.shape[1] != d2)
                throw std::runtime_error("Input shape must match n and d2");
            return new BC_Dual(n, d, d2, static_cast<float*>(buf.ptr), static_cast<float*>(buf2.ptr));
        }))
        .def("dkmips_avg", [](BC_Dual& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_avg(k, lambda, c, query_ptr);

            py::list pyResult;  
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }

            delete[] result;
            return pyResult;
        })
        .def("dkmips_max", [](BC_Dual& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_max(k, lambda, c, query_ptr);

            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }

            delete[] result;
            return pyResult;
        })
        .def("dkmips_avg_i2v", [](BC_Dual& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_avg_i2v(k, lambda, c, query_ptr);

            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }

            delete[] result;
            return pyResult;
        })
        .def("dkmips_max_i2v", [](BC_Dual& self, int k, float lambda, float c, py::array_t<float> query) {
            py::buffer_info buf = query.request();
            float* query_ptr = static_cast<float*>(buf.ptr);
            int* result = self.dkmips_max_i2v(k, lambda, c, query_ptr);

            py::list pyResult;
            for (int i = 0; i < k; i++) {
                pyResult.append(result[i]);
            }

            delete[] result;
            return pyResult;
        });
}