#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>
#include <CGAL/boost/graph/graph_traits_Polyhedron_3.h>
#include <CGAL/IO/OFF_reader.h>

#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>

#include <vector>
#include <fstream>
#include <iostream>


#include <CGAL/Surface_mesh.h>
#include <boost/foreach.hpp>

template< typename Polyhedron , typename Kernel >
bool fix_polyhedron(const char* filename, Polyhedron& polyhedron) {
    typedef typename Kernel::Point_3 Point;
    std::ifstream input(filename);
  
    if (!input) {
        std::cerr << "Cannot open file " << std::endl;
        return false;
    }
    
    std::vector<Point> points;
    std::vector< std::vector<std::size_t> > polygons;
    if (!CGAL::read_OFF(input, points, polygons)) {
        std::cerr << "Error parsing the OFF file " << std::endl;
        return false;
    }
    
    CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygons);
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(
        points,
        polygons,
        polyhedron
    );
  
    if (CGAL::is_closed(polyhedron) && 
        (!CGAL::Polygon_mesh_processing::is_outward_oriented(polyhedron))) {
        std::cout<< "reverse_face_orientation"<< std::endl;
        CGAL::Polygon_mesh_processing::reverse_face_orientations(polyhedron); 
    }
    
    return true;
}






