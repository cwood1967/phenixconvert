base_dir=getArgument;

IJ.log(base_dir);
run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory=["+ base_dir +"] layout_file=gc_tile_config.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]");

saveAs("Tiff", base_dir + ".tif");

run("Close All");
run("Quit");