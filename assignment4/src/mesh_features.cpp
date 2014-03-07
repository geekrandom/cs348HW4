#include "mesh_features.h"
using namespace OpenMesh;

bool isSilhouette(Mesh &mesh, const Mesh::EdgeHandle &e, Vec3f cameraPos)  {
	// CHECK IF e IS A SILHOUETTE HERE -----------------------------------------------------------------------------
    //get halfedges
    Mesh::HalfedgeHandle h0 = mesh.halfedge_handle(e,0);
    Mesh::HalfedgeHandle h1 = mesh.halfedge_handle(e,1);

    //Calculate view from the vertex
    Vec3f vertex(mesh.point(mesh.from_vertex_handle(h0)));
    Vec3f viewRay = vertex - cameraPos;
    
    //Normals give information about triangle views
    Vec3f normal0(mesh.normal(mesh.face_handle(h0)));
    Vec3f normal1(mesh.normal(mesh.face_handle(h1)));
    
    float d0 = normal0[0]*viewRay[0] + normal0[1]*viewRay[1] + normal0[2]*viewRay[2];
    float d1 = normal1[0]*viewRay[0] + normal1[1]*viewRay[1] + normal1[2]*viewRay[2];
    
    //return d0*d1 <= 0;

    if(d0*d1 <= 0) return true;    

	// -------------------------------------------------------------------------------------------------------------
    return false;
}

bool isSharpEdge(Mesh &mesh, const Mesh::EdgeHandle &e) {
	// CHECK IF e IS SHARP HERE ------------------------------------------------------------------------------------

    Mesh::HalfedgeHandle h0 = mesh.halfedge_handle(e,0);
    Mesh::HalfedgeHandle h1 = mesh.halfedge_handle(e,1);
    
    Vec3f n0(mesh.normal(mesh.face_handle(h0)));
    Vec3f n1(mesh.normal(mesh.face_handle(h1)));
    
    float dot = n0[0]*n1[0] + n0[1]*n1[1] + n0[2]*n1[2];
    
    if (dot <= .5) return true;

    //return dot <= .5;

	// -------------------------------------------------------------------------------------------------------------

    return false;
}

bool isFeatureEdge(Mesh &mesh, const Mesh::EdgeHandle &e, Vec3f cameraPos) {
	return mesh.is_boundary(e) || isSilhouette(mesh,e, cameraPos) || isSharpEdge(mesh,e);
}

