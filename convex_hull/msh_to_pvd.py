from dolfin import File, Mesh

mesh = Mesh("brain.xml")
File("test.pvd") << mesh
