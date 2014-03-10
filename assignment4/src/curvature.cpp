#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>
#include "curvature.h"
using namespace OpenMesh;
using namespace Eigen;
using namespace std;

#define esp 0.0001


//Find the K values and T values from the eigenvectors from the matrix
void FindKTFromMatrix(Matrix3d m, CurvatureInfo *info) {

        EigenSolver<Matrix3d> solver;
        solver.compute(m, true);
        
        Vector3d e1, e2, e3;
        e1 = solver.pseudoEigenvectors().block(0,0,3,1);
        e2 = solver.pseudoEigenvectors().block(0,1,3,1);
        e3 = solver.pseudoEigenvectors().block(0,2,3,1);
        
        double v1, v2, v3;
        v1 = real(solver.eigenvalues()(0));
        v2 = real(solver.eigenvalues()(1));
        v3 = real(solver.eigenvalues()(2));

        
        //T1, T2, k1, k2 corresponding to algorithm in paper
        Vector3d T1, T2;
        double k1, k2;

        //find the eigen value == 0
        //thus removing the normal vector
        if(v1 < esp && v1 > -esp) {
            T1 = e2;
            T2 = e3;
            k1 = v2;
            k2 = v3;
        } else if (v2 < esp && v2 > -esp) {
            T1 = e1;
            T2 = e3;
            k1 = v1;
            k2 = v3;
        } else if (v3 < esp && v3 > -esp) {
            T1 = e1;
            T2 = e2;
            k1 = v1;
            k2 = v2;
        }
        
		// In the end you need to fill in this struct
		info->curvatures[0] = k1;
		info->curvatures[1] = k2;
		info->directions[0] = Vec3f(T1[0], T1[1], T1[2]);
		info->directions[1] = Vec3f(T2[0], T2[1], T2[2]); 
          
}


void computeCurvature(Mesh &mesh, OpenMesh::VPropHandleT<CurvatureInfo> &curvature) {
	for (Mesh::VertexIter it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it) {
		// WRITE CODE HERE TO COMPUTE THE CURVATURE AT THE CURRENT VERTEX
        // ----------------------------------------------
		// -----------------------------------------------------------------------
        //code from Assignment 3

		Vec3f normal = mesh.normal(it.handle());
		Vector3d N(normal[0],normal[1],normal[2]); 
        // example of converting to Eigen's vector class for easier math

        Vec3f meshVi = mesh.point(it.handle());
        Vector3d vi(meshVi[0], meshVi[1], meshVi[2]);

        double TotalArea = 0;

        Mesh::VertexOHalfedgeIter area_iter;
        area_iter = mesh.voh_iter(it.handle());

        for(; area_iter; ++area_iter) {
            TotalArea += mesh.calc_sector_area(area_iter.handle());
        }

        Matrix3d m = Matrix3d::Zero();
    
        // circulate around the current vertex
        Mesh::VertexOHalfedgeIter he_iter = mesh.voh_iter(it.handle());

        //Iterate over all the surrounding vertices and calculate their contribution to M
        for(; he_iter; ++he_iter) {
            OpenMesh::HalfedgeHandle heHandle = he_iter.current_halfedge_handle();
            Vec3f meshVj = mesh.point(mesh.to_vertex_handle(heHandle));
            Vector3d vj(meshVj[0], meshVj[1], meshVj[2]); 

            Vector3d Tij = (Matrix3d::Identity(3, 3) - N * N.transpose()) * (vi - vj);
            Tij.normalize();

            Vector3d vjminvi = vj - vi;
            double constant = 2.0/(vjminvi.norm() * vjminvi.norm());
            double Kij = constant * N.dot(vj - vi);
            double A1 = mesh.calc_sector_area(heHandle);
            double A2 = mesh.calc_sector_area(mesh.opposite_halfedge_handle(heHandle));

            double Wij = (A1 + A2) / TotalArea;

            m(0,0) += Wij * Kij * Tij[0] * Tij[0];
            m(1,0) += Wij * Kij * Tij[0] * Tij[1];
            m(2,0) += Wij * Kij * Tij[0] * Tij[2];
            m(0,1) += Wij * Kij * Tij[1] * Tij[0];
            m(1,1) += Wij * Kij * Tij[1] * Tij[1];
            m(2,1) += Wij * Kij * Tij[1] * Tij[2];
            m(0,2) += Wij * Kij * Tij[2] * Tij[0];
            m(1,2) += Wij * Kij * Tij[2] * Tij[1];
            m(2,2) += Wij * Kij * Tij[2] * Tij[2];
        }
        
		CurvatureInfo info;
        FindKTFromMatrix(m, &info);

		mesh.property(curvature,it) = info;
	}
}

void computeViewCurvature(Mesh &mesh, OpenMesh::Vec3f camPos, \
    OpenMesh::VPropHandleT<CurvatureInfo> &curvature, \
    OpenMesh::VPropHandleT<double> &viewCurvature, \
    OpenMesh::FPropHandleT<OpenMesh::Vec3f> &viewCurvatureDerivative, \
    OpenMesh::VPropHandleT<double> &viewCurvaturePerp, \
    OpenMesh::FPropHandleT<OpenMesh::Vec3f> &viewCurvaturePerpDerivative) {
	// WRITE CODE HERE TO COMPUTE CURVATURE IN THE VIEW PROJECTION PROJECTED ON THE TANGENT PLANE 
	// Compute vector to viewer and project onto tangent plane, then use components in principal directions to find curvature
    for (Mesh::VertexIter it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it) {
        Vec3f vertex = mesh.point(it.handle());

        Vector3d view(camPos[0]-vertex[0],camPos[1]-vertex[1],camPos[2]-vertex[2]);
        view = view.normalized();

        //Calculate view curvature
        double k1 = mesh.property(curvature,it).curvatures[0];
        double k2 = mesh.property(curvature,it).curvatures[1];
        Vec3f pd1 = mesh.property(curvature,it).directions[0];
        Vec3f pd2 = mesh.property(curvature,it).directions[1];
        Vector3d direction1(pd1[0],pd1[1],pd1[2]);
        Vector3d direction2(pd2[0],pd2[1],pd2[2]);
        double u = direction1.dot(view);
        double v = direction2.dot(view);
   
        double Tg = (k2-k1) * u * v;
        double Kn = k1*u*u + k2*v*v;
    
        mesh.property(viewCurvaturePerp,it) = Tg;
        mesh.property(viewCurvature,it) = Kn;
    }

	// --------------------------------
	// We'll use the finite elements piecewise hat method to find per-face gradients of the view curvature
	// CS 348a doesn't cover how to differentiate functions on a mesh (Take CS 468! Spring 2013!) so we provide code here
	
	for (Mesh::FaceIter it = mesh.faces_begin(); it != mesh.faces_end(); ++it) {
		double c[3];
		double Tg[3];
		Vec3f p[3];
		
		Mesh::ConstFaceVertexIter fvIt = mesh.cfv_iter(it);
		for (int i = 0; i < 3; i++) {
			p[i] = mesh.point(fvIt.handle());
			c[i] = mesh.property(viewCurvature,fvIt.handle());
			Tg[i] = mesh.property(viewCurvaturePerp,fvIt.handle());
			++fvIt;
		}
		
		Vec3f N = mesh.normal(it.handle());
		double area = mesh.calc_sector_area(mesh.halfedge_handle(it.handle()));

		mesh.property(viewCurvatureDerivative,it) = (N%(p[0]-p[2]))*(c[1]-c[0])/(2*area) + (N%(p[1]-p[0]))*(c[2]-c[0])/(2*area);
		mesh.property(viewCurvaturePerpDerivative,it) = (N%(p[0]-p[2]))*(Tg[1]-Tg[0])/(2*area) + (N%(p[1]-p[0]))*(Tg[2]-Tg[0])/(2*area);
	}
}

