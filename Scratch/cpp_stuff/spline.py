import dolfin as df

cpp_code = """

namespace dolfin {

class MyFunc : public Expression {
    std::vector<double> *time_array;
    std::vector<double> *value_array;
    std::vector< std::vector<double> > *point_list;
    double time;

    public:
        MyFunc() : Expression() {
        }

        void eval(Array<double> &value, const Array<double> &x) const {

            std::vector<double> square_dist(point_list->size());
            double tmp_sum;

            std::cout << point_list->size() << std::endl;            

            //for (std::size_t ii = 0; ii < point_list->size(); ++ii) {
            //    tmp_sum = 0;
            //    for (std::size_t j = 0; j < 3; j++) {
            //        tmp_sum += x[j] - (*point_list)[ii][j];
            //    }
            //    square_dist[ii] = tmp_sum*tmp_sum;
            //}

            //double sqdist_sum = 0;
            //for (auto e: square_dist) {
            //    sqdist_sum += e;
            //}

            double tmp_val = 0;

            //for (std::size_t i = 0; i < time_array->size() - 1; i++) {
            //    if ((*time_array)[i] < time && time < (*time_array)[i + 1]) {
            //    tmp_val += (*value_array)[i];
            //        tmp_val += (*value_array)[i] +
            //                 (time - (*time_array)[i])*
            //                 ((*value_array)[i + 1] - (*value_array)[i])/
            //                 ((*time_array)[i + 1] - (*time_array)[i])
            //                 *square_dist[i]/sqdist_sum;
            //    }
            //}
            value[0] = tmp_val;
        }
};
}
"""

mesh = df.UnitSquareMesh(10, 10)
V = df.FunctionSpace(mesh, "CG", 1)

import numpy as np
time = np.linspace(0, 1, 5)
value = np.arange(5)
point_list = np.arange(9).reshape(3, 3)

my_expr = df.Expression(cppcode=cpp_code, degree=1)
my_expr.time_array = time
my_expr.value_array = value
my_expr.point_list = point_list


foo = df.interpolate(my_expr, V)
