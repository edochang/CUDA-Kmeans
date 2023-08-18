#include <io.h>

void read_file(struct options_t* args,
               double**          points) {

  	// Open file
	std::ifstream in;
	in.open(args->input_filename);
	// Get num vals
	in >> args->records;

	// Alloc input and output arrays.  Each input record contains a record ID so it's dims + 1 elements.
	int records_dims = args->records * (args->dims + 1);
	*points = (double *)malloc(records_dims * sizeof(double));

	//printf("io: records_dims: %d \n", records_dims);  // debug statement

	// Read input vals
	double temp;
	for (int i = 0; i < records_dims; ++i) {
		in >> temp;
		
		//if (i % 17 == 0) {  printf("io: in: %lf \n", temp); }  // debug statement
		//printf("io: in: %.15f \n", temp);  // debug statement

		(*points)[i] = temp;
	}
}

/*
void write_file(struct options_t*         args,
               	struct prefix_sum_args_t* opts) {
  // Open file
	std::ofstream out;
	out.open(args->out_file, std::ofstream::trunc);

	// Write solution to output file
	for (int i = 0; i < opts->n_vals; ++i) {
		out << opts->output_vals[i] << std::endl;
		//printf("io: Writing from i(%d) the value: %d \n", i, opts->output_vals[i]); // debug statement
	}

	out.flush();
	out.close();
	
	// Free memory
	free(opts->input_vals);
	free(opts->output_vals);
}
*/