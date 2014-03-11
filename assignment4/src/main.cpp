#include <OpenMesh/Core/IO/Options.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "curvature.h"
#include "mesh_features.h"
#include "image_generation.h"
#include "decimate.h"
#include "shader.h"
using namespace std;
using namespace OpenMesh;
using namespace Eigen;


#define dervThresContour .1
#define dervThresHighlight 80
#define angThres 0.2
#define curveEsp .001
#define threshPH 0.3

VPropHandleT<double> viewCurvature;
FPropHandleT<Vec3f> viewCurvatureDerivative;
VPropHandleT<CurvatureInfo> curvature;
Mesh mesh;
Shader *shaderToon;

bool leftDown = false, rightDown = false, middleDown = false;
int lastPos[2];
float cameraPos[4] = {0,0,4,1};
Vec3f up, pan;
int windowWidth = 640, windowHeight = 480;
bool showSurface = true, showAxes = false, showCurvature = false, showNormals = false;
bool showContours = true, showSHighlights = true, showPHighlights = true;

float specular[] = { 1.0, 1.0, 1.0, 1.0 };
float shininess[] = { 50.0 };

Vector3d meshVecToCompVec(Vec3f vectorIn){
    return Vector3d(vectorIn[0], vectorIn[1], vectorIn[2]);
}

void draw_segment_ridge(Vector3d v0, Vector3d v1, Vector3d v2,
			double emax0, double emax1, double emax2,
			double kmax0, double kmax1, double kmax2){
	// Interpolate to find ridge/valley line segment endpoints
	// in this triangle and the curvatures there
	double w10 = abs(emax0) / (abs(emax0) + abs(emax1));
	double w01 = 1.0 - w10;
	Vector3d p01 = w01 * v0 + w10 * v1;
	double k01 = abs(w01 * kmax0 + w10 * kmax1);

	// Connect first point to second one (on next edge)
	double w21 = abs(emax1) / (abs(emax1) + abs(emax2));
	double w12 = 1.0 - w21;
	Vector3d p12 = w12 * v1 + w21 * v2;
	double k12 = abs(w12 * kmax1 + w21 * kmax2);

	// Don't draw below threshold
	k01 -= threshPH;
	if (k01 < 0.0)
		k01 = 0.0;
	k12 -= threshPH;
	if (k12 < 0.0)
		k12 = 0.0;

	// Skip lines that you can't see...
	if (k01 == 0.0 && k12 == 0.0)
		return;

	// Fade lines
	k01 /= (k01 + threshPH);
	k12 /= (k12 + threshPH);

	// Draw the line segment
	glColor4f(.95, .95, .95, 1);
	glVertex3f(p01[0], p01[1], p01[2]);
	glVertex3f(p12[0], p12[1], p12[2]);
}


void renderSuggestive(Vec3f actualCamPos, bool renderContours) { 
    if(renderContours){
        glColor3f(0,0,0);
    }else{
        glColor3f(1,1,1);
    }
    // use this camera position to account for panning etc.
	// RENDER SUGGESTIVE CONTOURS HERE
    // ----------------------------------------------------
    for (Mesh::ConstFaceIter f_it=mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        Vec3f deriv = mesh.property(viewCurvatureDerivative, f_it);
        Vector3d derivative(deriv[0], deriv[1], deriv[2]);
        Mesh::FaceVertexIter fv_it =mesh.fv_iter(f_it);
        
        float c1 =mesh.property(viewCurvature, fv_it);
        Vec3f v1 =mesh.point(fv_it.handle());
        float c2 =mesh.property(viewCurvature, ++fv_it);
        Vec3f v2 =mesh.point(fv_it.handle());
        float c3 =mesh.property(viewCurvature, ++fv_it);
        Vec3f v3 =mesh.point(fv_it.handle());
        
        std::vector<Vec3f> edges;
        // checking if the change in sign appears along which edge.
        if (c1*c2 <= curveEsp) {
            double sum = abs(c2) + abs(c1);
            Vec3f curvPoint12 = (v1 * abs(c1) + v2 * abs(c2)) / sum;
            edges.push_back(curvPoint12);
        }
        if (c1*c3 <= curveEsp) {
            double sum = abs(c3) + abs(c1);
            Vec3f curvPoint13 = (v1 * abs(c1) + v3 * abs(c3)) / sum;
            edges.push_back(curvPoint13);
        }
        if (c2*c3 <= curveEsp) {
            double sum = abs(c2) + abs(c3);
            Vec3f curvPoint23 = (v3 * abs(c3) + v2 * abs(c2)) / sum;
            edges.push_back(curvPoint23);
        }
        if (edges.size() == 2) {
            Vec3f fCentroid = edges[0]+edges[1];
            fCentroid = ((float)1.0/2)*fCentroid;
            
            //the view of the triangle
            Vector3d view(actualCamPos[0]-fCentroid[0], actualCamPos[1]-fCentroid[1], \
                actualCamPos[2]-fCentroid[2]);
            view = view.normalized();

            Vec3f normal = mesh.normal(f_it.handle());
            Vector3d surfaceNorm(normal[0], normal[1], normal[2]);
            //surfaceNorm = surfaceNorm.normalized();

            Vector3d w = view - surfaceNorm * view.dot(surfaceNorm);
            double wNorm = w.norm();
            double DwKrOverwNorm = -derivative.dot(w)/(wNorm);

            if((renderContours && DwKrOverwNorm > dervThresContour) || \
            (!renderContours && DwKrOverwNorm < -dervThresHighlight)) {
                float cosine = surfaceNorm.dot(view);
                //between desired legal angles
                if(cosine > angThres) {
                    //different curvture directions
                    //inflection point
                    glVertex3f(edges[0][0], edges[0][1], edges[0][2]);
                    glVertex3f(edges[1][0], edges[1][1], edges[1][2]);
                }
            }
        }
    }
	// ----------------------------------------------------------------------
}

void renderPrincipalHighlights(Vec3f actualCamPos) { 
    glColor3f(.9,.9,.9);

    for (Mesh::ConstFaceIter f_it=mesh.faces_begin(); \
    f_it != mesh.faces_end(); ++f_it) {
        Mesh::FaceVertexIter fv_it =mesh.fv_iter(f_it);
        
        Vector3d v0 = meshVecToCompVec(mesh.point(fv_it.handle()));
        Vector3d n0 = meshVecToCompVec(mesh.normal(fv_it.handle()));
        double k10 = mesh.property(curvature,fv_it.handle()).curvatures[0];
        double k20 = mesh.property(curvature,fv_it.handle()).curvatures[1];
        Vector3d d0 = meshVecToCompVec(mesh.property(curvature, \
            fv_it.handle()).directions[0]);
        ++fv_it;

        Vector3d v1 = meshVecToCompVec(mesh.point(fv_it.handle()));
        Vector3d n1 = meshVecToCompVec(mesh.normal(fv_it.handle()));
        double k11 = mesh.property(curvature,fv_it.handle()).curvatures[0];
        double k21 = mesh.property(curvature,fv_it.handle()).curvatures[1];
        Vector3d d1 = meshVecToCompVec(mesh.property(curvature, \
            fv_it.handle()).directions[0]);
        ++fv_it;

        Vector3d v2 = meshVecToCompVec(mesh.point(fv_it.handle()));
        Vector3d n2 = meshVecToCompVec(mesh.normal(fv_it.handle()));
        double k12 = mesh.property(curvature,fv_it.handle()).curvatures[0];
        double k22 = mesh.property(curvature,fv_it.handle()).curvatures[1];
        Vector3d d2 = meshVecToCompVec(mesh.property(curvature, \
            fv_it.handle()).directions[0]);


	    double kmax = abs(k10);
        // dref is the e1 vector with the largest |k1|
	    Vector3d dref = d0;
	    if (abs(k11) > kmax)
		    kmax = abs(k11), dref = d1;
	    if (abs(k12) > kmax)
		    kmax = abs(k12), dref = d2;

        if(d0.dot(dref) < 0) d0 = -d0;
        if(d1.dot(dref) < 0) d1 = -d1;
        if(d2.dot(dref) < 0) d2 = -d2;
        
        // If directions have flipped (more than 45 degrees), then give up
        if ((d0.dot(dref)) < M_SQRT1_2 ||
            (d1.dot(dref)) < M_SQRT1_2 ||
            (d2.dot(dref)) < M_SQRT1_2)
          continue;

        Vector3d viewdir0 = meshVecToCompVec(actualCamPos) - v0;
        Vector3d viewdir1 = meshVecToCompVec(actualCamPos) - v1;
        Vector3d viewdir2 = meshVecToCompVec(actualCamPos) - v2;

        viewdir0 = viewdir0.normalized();
        viewdir1 = viewdir1.normalized();
        viewdir2 = viewdir2.normalized();
    
        double dot0 = viewdir0.dot(d0);
        double dot1 = viewdir1.dot(d1);
        double dot2 = viewdir2.dot(d2);

	    // We have a "zero crossing" if the dot products along an edge
	    // have opposite signs
	    int z01 = (dot0*dot1 <= 0.0);
	    int z12 = (dot1*dot2 <= 0.0);
	    int z20 = (dot2*dot0 <= 0.0);

	    if (z01 + z12 + z20 < 2)
		    return;

        double test0 = (k10 - k20)*(k10 - k20) * viewdir0.dot(n0);
        double test1 = (k11 - k21)*(k11 - k21) * viewdir1.dot(n1);
        double test2 = (k12 - k22)*(k12 - k22) * viewdir2.dot(n2);

        if(!z01){
		    draw_segment_ridge(v1, v2, v0,
		       dot1, dot2, dot0,
		       test1, test2, test0);
        }else if(!z12){
    		draw_segment_ridge(v2, v0, v1,
			   dot2, dot0, dot1,
			   test2, test0, test1);
        }else if(!z20){
    		draw_segment_ridge(v0, v1, v2,
			   dot0, dot1, dot2,
			   test0, test1, test2);
        }
    }
}

void renderTriMesh(){
	// WRITE CODE HERE TO RENDER THE TRIANGLES OF THE MESH  
    //---------------------------------------------------------
    std::vector<unsigned int> indices;
      indices.clear();
      indices.reserve(mesh.n_faces()*3);

    for(Mesh::FaceIter f_it = mesh.faces_begin(); f_it !=
                mesh.faces_end(); ++f_it) {
      
        Mesh::ConstFaceVertexIter cfv_it;
        cfv_it =mesh.cfv_iter(f_it.handle());

        indices.push_back(cfv_it.handle().idx());
        indices.push_back((++cfv_it).handle().idx());
        indices.push_back((++cfv_it).handle().idx());
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, mesh.points());
    glNormalPointer(GL_FLOAT, 0, mesh.vertex_normals());

    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, &indices[0]);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
	// -------------------------------------------------------------
}

void renderNormals(){
	glBegin(GL_LINES);
	glColor3f(0,1,0);
	for (Mesh::ConstVertexIter it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it) {
		Vec3f n = mesh.normal(it.handle());
		Vec3f p = mesh.point(it.handle());
		Vec3f d = p + n*.01;
		glVertex3f(p[0],p[1],p[2]);
		glVertex3f(d[0],d[1],d[2]);
	}
	glEnd();
}

void renderContours(Vec3f actualCamPos){
	// We'll be nice and provide you with code to render feature edges below
	glBegin(GL_LINES);
	glColor3f(0,0,0);
	glLineWidth(1.5f);
	for (Mesh::ConstEdgeIter it = mesh.edges_begin(); it != mesh.edges_end(); ++it)
		if (isFeatureEdge(mesh,*it,actualCamPos)) {
			Mesh::HalfedgeHandle h0 = mesh.halfedge_handle(it,0);
			Mesh::HalfedgeHandle h1 = mesh.halfedge_handle(it,1);
			Vec3f source(mesh.point(mesh.from_vertex_handle(h0)));
			Vec3f target(mesh.point(mesh.from_vertex_handle(h1)));
			glVertex3f(source[0],source[1],source[2]);
			glVertex3f(target[0],target[1],target[2]);
		}
	glEnd();
}

void renderCurvature(){
	// WRITE CODE HERE TO RENDER THE PRINCIPAL DIRECTIONS YOU COMPUTED 
    //---------------------------------------------
	glBegin(GL_LINES);
	for (Mesh::ConstVertexIter it = mesh.vertices_begin(); it != mesh.vertices_end(); ++it) {
		Vec3f p = mesh.point(it.handle());
		CurvatureInfo info = mesh.property(curvature, it.handle());

        Vec3f Tone = p + info.directions[0]*.01;
        Vec3f Ttwo = p + info.directions[1]*.01;
        Vec3f Pone = p - info.directions[0]*.01;
        Vec3f Ptwo = p - info.directions[1]*.01;
		glColor3f(1,0,0);
		glVertex3f(Pone[0],Pone[1],Pone[2]);
		glVertex3f(Tone[0],Tone[1],Tone[2]);
		glColor3f(0,0,1);
		glVertex3f(Ptwo[0],Ptwo[1],Ptwo[2]);
		glVertex3f(Ttwo[0],Ttwo[1],Ttwo[2]);
	}
	glEnd();
	// -----------------------------------------------------------------
}












void renderMesh() {
    if (!showSurface) glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE); 
    // render regardless to remove hidden lines
	glUseProgram(shaderToon->programID());

	glEnable(GL_LIGHTING);
	Vec3f actualCamPos(cameraPos[0]+pan[0],cameraPos[1]+pan[1],cameraPos[2]+pan[2]);
    GLfloat glCamera[] = {actualCamPos[0]+2.5,actualCamPos[1]+2.5,actualCamPos[2]+2.5, 1 };
	glLightfv(GL_LIGHT0, GL_POSITION, glCamera);

	glDepthRange(0.001,1);
	glEnable(GL_NORMALIZE);
	
	renderTriMesh();
	if (!showSurface) glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
	
	glUseProgram(0);
	glDisable(GL_LIGHTING);
	glDepthRange(0,0.999);
	
    glBegin(GL_LINES);
    glLineWidth(1.5);
	if(showContours) renderSuggestive(actualCamPos, true);
    if(showSHighlights)	renderSuggestive(actualCamPos, false);
    if(showPHighlights)	renderPrincipalHighlights(actualCamPos);
    glEnd();
		
    renderContours(actualCamPos);

	if (showCurvature) renderCurvature();
	if (showNormals) renderNormals();
	
	glDepthRange(0,1);
}


void display() {
	glClearColor(1,1,1,1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glShadeModel(GL_SMOOTH);
	glMaterialfv(GL_FRONT, GL_SPECULAR, specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, shininess);
	glEnable(GL_LIGHT0);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0,0,windowWidth,windowHeight);
	
	float ratio = (float)windowWidth / (float)windowHeight;
	gluPerspective(50, ratio, 1, 1000); // 50 degree vertical viewing angle, zNear = 1, zFar = 1000
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(cameraPos[0]+pan[0], cameraPos[1]+pan[1], cameraPos[2]+pan[2], \
        pan[0], pan[1], pan[2], up[0], up[1], up[2]);
	
	// Draw mesh
	renderMesh();

	// Draw axes
	if (showAxes) {
		glDisable(GL_LIGHTING);
		glBegin(GL_LINES);
		glLineWidth(1);
			glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0); // x axis
			glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0); // y axis
			glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1); // z axis
		glEnd(/*GL_LINES*/);
	}
	
	glutSwapBuffers();
}

void mouse(int button, int state, int x, int y) {
	if (button == GLUT_LEFT_BUTTON) leftDown = (state == GLUT_DOWN);
	else if (button == GLUT_RIGHT_BUTTON) rightDown = (state == GLUT_DOWN);
	else if (button == GLUT_MIDDLE_BUTTON) middleDown = (state == GLUT_DOWN);
	
	lastPos[0] = x;
	lastPos[1] = y;
}

void mouseMoved(int x, int y) {
	int dx = x - lastPos[0];
	int dy = y - lastPos[1];
	Vec3f curCamera(cameraPos[0],cameraPos[1],cameraPos[2]);
	Vec3f curCameraNormalized = curCamera.normalized();
	Vec3f right = up % curCameraNormalized;

	if (leftDown) {
		// Assume here that up vector is (0,1,0)
		Vec3f newPos = curCamera - 2*(float)((float)dx/(float)windowWidth) * right + 2*(float)((float)dy/(float)windowHeight) * up;
		newPos = newPos.normalized() * curCamera.length();
		
		up = up - (up | newPos) * newPos / newPos.sqrnorm();
		up.normalize();
		
		for (int i = 0; i < 3; i++) cameraPos[i] = newPos[i];
	}
	else if (rightDown) for (int i = 0; i < 3; i++) cameraPos[i] *= pow(1.1,dy*.1);
	else if (middleDown) {
		pan += -2*(float)((float)dx/(float)windowWidth) * right + 2*(float)((float)dy/(float)windowHeight) * up;
	}

	
	lastPos[0] = x;
	lastPos[1] = y;
	
	Vec3f actualCamPos(cameraPos[0]+pan[0],cameraPos[1]+pan[1],cameraPos[2]+pan[2]);
	computeViewCurvature(mesh,actualCamPos,curvature,viewCurvature,viewCurvatureDerivative);
	
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
	Vec3f actualCamPos(cameraPos[0]+pan[0],cameraPos[1]+pan[1],cameraPos[2]+pan[2]);

	if (key == 's' || key == 'S') showSurface = !showSurface;
	else if (key == 'h' || key == 'H') showSHighlights = !showSHighlights;
	else if (key == 'p' || key == 'P') showPHighlights = !showPHighlights;
	else if (key == 't' || key == 'T') showContours = !showContours;
	else if (key == 'a' || key == 'A') showAxes = !showAxes;
	else if (key == 'c' || key == 'C') showCurvature = !showCurvature;
	else if (key == 'n' || key == 'N') showNormals = !showNormals;
	else if (key == 'w' || key == 'W') writeImage(mesh, windowWidth, \
        windowHeight, "renderedImage.svg", actualCamPos);
	else if (key == 'q' || key == 'Q') exit(0);
	glutPostRedisplay();
}

void reshape(int width, int height) {
	windowWidth = width;
	windowHeight = height;
	glutPostRedisplay();
}

int main(int argc, char** argv) {
	if (argc < 2) {
		cout << "Usage: " << argv[0] << " mesh_filename\n";
		exit(0);
	}
	
	IO::Options opt;
	opt += IO::Options::VertexNormal;
	opt += IO::Options::FaceNormal;
	
	mesh.request_face_normals();
	mesh.request_vertex_normals();
	
	cout << "Reading from file " << argv[1] << "...\n";
	if ( !IO::read_mesh(mesh, argv[1], opt )) {
		cout << "Read failed.\n";
		exit(0);
	}

	cout << "Mesh stats:\n";
	cout << '\t' << mesh.n_vertices() << " vertices.\n";
	cout << '\t' << mesh.n_edges() << " edges.\n";
	cout << '\t' << mesh.n_faces() << " faces.\n";
	
	//simplify(mesh,.1f);
	
	mesh.update_normals();
	
	mesh.add_property(viewCurvature);
	mesh.add_property(viewCurvatureDerivative);
	mesh.add_property(curvature);
	
	// Move center of mass to origin
	Vec3f center(0,0,0);
	for (Mesh::ConstVertexIter vIt = mesh.vertices_begin(); vIt != mesh.vertices_end(); ++vIt) center += mesh.point(vIt);
	center /= mesh.n_vertices();
	for (Mesh::VertexIter vIt = mesh.vertices_begin(); vIt != mesh.vertices_end(); ++vIt) mesh.point(vIt) -= center;

	// Fit in the unit sphere
	float maxLength = 0;
	for (Mesh::ConstVertexIter vIt = mesh.vertices_begin(); vIt != mesh.vertices_end(); ++vIt) maxLength = max(maxLength, mesh.point(vIt).length());
	for (Mesh::VertexIter vIt = mesh.vertices_begin(); vIt != mesh.vertices_end(); ++vIt) mesh.point(vIt) /= maxLength;
	
	computeCurvature(mesh,curvature);

	up = Vec3f(0,1,0);
	pan = Vec3f(0,0,0);
	
	Vec3f actualCamPos(cameraPos[0]+pan[0],cameraPos[1]+pan[1],cameraPos[2]+pan[2]);
	computeViewCurvature(mesh,actualCamPos,curvature,viewCurvature,viewCurvatureDerivative);

	glutInit(&argc, argv); 
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); 
	glutInitWindowSize(windowWidth, windowHeight); 
	glutCreateWindow(argv[0]);

	glutDisplayFunc(display);
	glutMotionFunc(mouseMoved);
	glutMouseFunc(mouse);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);

    GLint error = glewInit(); 
    cout << glGetString(GL_VERSION) << std::endl;;
    if (GLEW_OK != error) {
        std::cerr << glewGetErrorString(error) << std::endl;
        exit(-1);
    }
    if (!GLEW_VERSION_2_0) {
        std::cerr << "This program requires OpenGL 2.0" << std::endl;
        exit(-1);
    }
    
    shaderToon = new Shader("shaders/toon");
	if (!shaderToon->loaded()) {
		std::cerr << "Shader failed to load" << std::endl;
		std::cerr << shaderToon->errors() << std::endl;
		exit(-1);
	}

	glutMainLoop();
	
	return 0;
}
