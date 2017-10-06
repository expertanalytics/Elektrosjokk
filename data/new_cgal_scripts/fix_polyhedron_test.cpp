#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Surface_mesh.h>


#include <boost/function_output_iterator.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#include <CGAL/Polyhedron_items_with_id_3.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/boost/graph/graph_traits_Polyhedron_3.h>
#include <CGAL/boost/graph/graph_traits_Surface_mesh.h>

#include "fixpolyhedron.h"

#include "edge_collapse.h"
// #include "hole_filling.h"       // Requries eigen 3.2 or later (manual installation)
#include "intersecting.h"
#include "simplify.h"
#include "sticthing.h"

// #include "remeshing.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> Mesh;


int main(int argc, char* argv[]) {
    const char* filename = argv[1];
    const char* outputfile = argv[2];
    Mesh mesh;
    if (fix_polyhedron<Mesh, K>(filename, mesh)) {
        double target_edge_length = 0.8;
        unsigned int nb_iter = 3;
        // edge_collapse(mesh);        // seems to improve stuff 
        //
        // fill_holes(mesh);        // Requires eigen 3.2 or later
        // triangulate_faces(mesh);     // requires eigen 3.2
        // sticthing(mesh);

        // simplify(mesh);      // This one is the sinner
        // remeshing();     // Fix this cgal 4.11
        // remeshing(mesh, target_edge_length, nb_iter, false);     // Fix this cgal 4.11
        std::ofstream out(outputfile);
        out << mesh;
        out.close();
        return 1;
    }

    return 0;
}
