#include "cgnslib.h"
#include <iostream>

int main(int argc, char* argv[]) {
    double x[9][17][21],y[9][17][21],z[9][17][21];
    cgsize_t isize[3][3];
    int ni,nj,nk,i,j,k;
    int index_file,icelldim,iphysdim,index_base;
    int index_zone,index_coord;
    char basename[33],zonename[33];

    /* create gridpoints for simple example: */
    ni=21;
    nj=17;
    nk=9;
    for (k=0; k < nk; ++k)
    {
        for (j=0; j < nj; ++j)
        {
        for (i=0; i < ni; ++i)
        {
            x[k][j][i]=i;
            y[k][j][i]=j;
            z[k][j][i]=k;
        }
        }
    }
    std::cout << "Created simple 3-D grid points" << std::endl;

    /* WRITE X, Y, Z GRID POINTS TO CGNS FILE */
    /* open CGNS file for write */
    if (cg_open("grid_c.cgns",CG_MODE_WRITE,&index_file)) cg_error_exit();
    /* create base (user can give any name) */
    strcpy(basename,"Base");
    icelldim=3;
    iphysdim=3;
    cg_base_write(index_file,basename,icelldim,iphysdim,&index_base);
    /* define zone name (user can give any name) */
    strcpy(zonename,"Zone  1");
    /* vertex size */
    isize[0][0]=21;
    isize[0][1]=17;
    isize[0][2]=9;
    /* cell size */
    isize[1][0]=isize[0][0]-1;
    isize[1][1]=isize[0][1]-1;
    isize[1][2]=isize[0][2]-1;
    /* boundary vertex size (always zero for structured grids) */
    isize[2][0]=0;
    isize[2][1]=0;
    isize[2][2]=0;
    /* create zone */
    cg_zone_write(index_file,index_base,zonename,*isize,Structured,&index_zone);
    /* write grid coordinates (user must use SIDS-standard names here) */
    cg_coord_write(index_file,index_base,index_zone,RealDouble,"CoordinateX",
        x,&index_coord);
    cg_coord_write(index_file,index_base,index_zone,RealDouble,"CoordinateY",
        y,&index_coord);
    cg_coord_write(index_file,index_base,index_zone,RealDouble,"CoordinateZ",
        z,&index_coord);
    /* close CGNS file */
    cg_close(index_file);
    std::cout << "Successfully wrote grid to file grid_c.cgns" << std::endl;
    
    return 0;
}