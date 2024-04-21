#include <iostream> // Biblioteca de entrada salida
#define _USE_MATH_DEFINES
#include <math.h>
#include <gl\freeglut.h> // Biblioteca grafica
#define TITULO "Estrella de David"
static GLint id;

void triangulos() {
	glClearColor(1, 1, 1, 1);

	id = glGenLists(1);
	glNewList(id, GL_COMPILE);
	glPushAttrib(GL_CURRENT_BIT);
	glColor3f(0.0, 0.0, 0.3);

	glBegin(GL_TRIANGLE_STRIP);
	for (int i = 0; i < 4; i++) {
		double angle = (1.0 + (i * 4) % 12) * M_PI / 6;
		glVertex3f(1.0 * cos(angle), 1.0 * sin(angle), 0.0);
		glVertex3f(0.7 * cos(angle), 0.7 * sin(angle), 0.0);
	}
	glEnd();

	glBegin(GL_TRIANGLE_STRIP);
	for (int i = 0; i < 4; i++) {
		double angle = (3.0 + (i * 4) % 12) * M_PI / 6;
		glVertex3f(1.0 * cos(angle), 1.0 * sin(angle), 0.0);
		glVertex3f(0.7 * cos(angle), 0.7 * sin(angle), 0.0);
	}
	glEnd();

	glPopAttrib();
	glEndList();
}
void display()
// Funcion de atencion al dibujo
{
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glCallList(id);
	glFlush();
}
void reshape(GLint w, GLint h)
// Funcion de atencion al redimensionamiento
{
}
void main(int argc, char** argv)
// Programa principal
{
	glutInit(&argc, argv); // Inicializacion de GLUT
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB); // Alta de buffers a usar
	glutInitWindowSize(400, 400); // Tamanyo inicial de la ventana
	glutInitWindowPosition(50, 600);
	glutCreateWindow(TITULO); // Creacion de la ventana con su titulo
	std::cout << TITULO << " running" << std::endl; // Mensaje por consola
	glutDisplayFunc(display); // Alta de la funcion de atencion a display
	glutReshapeFunc(reshape); // Alta de la funcion de atencion a reshape
	triangulos();
	glutMainLoop(); // Puesta en marcha del programa
}