/*
Input file names are missing. They should be inserted in the following order:
 - inner mesh
 - outer mesh
*/

Surface Loop(1) = {1};
Surface Loop(2) = {2};

Physical Surface(1) = {1};
Physical Surface(2) = {2};

Volume(5) = {1};
Volume(6) = {1, 2};

Physical Volume(7) = {5};
Physical Volume(8) = {6};
