//
// Demonstrates how to generate of a Conduit Mesh Blueprint 
// description of an AMReX Single Level dataset and render this
// data in situ using with ALPINE Ascent.
// 

#include <AMReX_Utility.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <AMReX_Conduit_Blueprint.H>

#include "PeleC.H"

#include <conduit/conduit.hpp>
#include <conduit/conduit_blueprint.hpp>
#include <conduit/conduit_relay.hpp>

#include <ascent.hpp>

#include "AMReX_buildInfo.H"

using std::string;
using namespace amrex;

using namespace conduit;
using namespace ascent;

	void 
PeleC::writeAscentPlotFile (int nstep, int last_plot)
{
	std::cout << " nstep " << nstep << std::endl;
	std::vector<std::pair<int,int> > plot_var_map;
	for (int typ = 0; typ < desc_lst.size(); typ++)
		for (int comp = 0; comp < desc_lst[typ].nComp();comp++)
			if (parent->isStatePlotVar(desc_lst[typ].name(comp)) &&
					desc_lst[typ].getType() == IndexType::TheCellType())
				plot_var_map.push_back(std::pair<int,int>(typ,comp));

	int n_data_items = plot_var_map.size();
	std::cout << " n_data_items " << n_data_items << std::endl;

	Real time = state[State_Type].curTime();

	// We combine all of the multifabs -- state, derived, etc -- into one
	// multifab -- plotMF.
	// NOTE: we are assuming that each state variable has one component,
	// but a derived variable is allowed to have multiple components.
	// "density" "xmom" "ymom" "zmom" "rho_E" "rho_e" "Temp" "rho_NC" "rho_O2" "rho_N2"
	int       cnt   = 0;
	const int nGrow = 0;
	MultiFab  plotMF(grids,dmap,1,nGrow,MFInfo(),Factory());
	//MultiFab  plotMF(grids,dmap,n_data_items,nGrow,MFInfo(),Factory());

	//
	// Cull data from state variables -- use no ghost cells.
	//
	MultiFab* this_dat = 0;
	// for (int i = 0; i < n_data_items; i++)
	for (int i = 6; i < 7; i++)
	{
		int typ  = plot_var_map[i].first;
		int comp = plot_var_map[i].second;
		this_dat = &state[typ].newData();
		MultiFab::Copy(plotMF,*this_dat,comp,cnt,1,nGrow);
		std::cout << " cnt " << typ << " " << comp << std::endl;
		cnt++;
	}

	/////////////////////////////
	// Setup Ascent
	/////////////////////////////
	// Create an instance of Ascent
	Ascent ascent;
	Node open_opts;

	// for the MPI case, provide the mpi comm
#ifdef BL_USE_MPI
	open_opts["mpi_comm"] = MPI_Comm_c2f(ParallelDescriptor::Communicator());
#endif

	ascent.open(open_opts); 
	///////////////////////////////////////////////////////////////////
	// Wrap our AMReX Mesh into a Conduit Mesh Blueprint Tree
	///////////////////////////////////////////////////////////////////

	// Write a plotfile of the current data and write a conduit blueprint
	// file as well

	// in Base/AMReX_PlotFileUtil.cpp

	///////////////////////////////////////////////////////////////////
	// Wrap our AMReX Mesh into a Conduit Mesh Blueprint Tree
	///////////////////////////////////////////////////////////////////
	conduit::Node bp_mesh;
  	SingleLevelToBlueprint( plotMF, {"Temperature"}, geom, time, nstep, bp_mesh);

	Node verify_info;
	if(!conduit::blueprint::mesh::verify(bp_mesh,verify_info))
	{
		// verify failed, print error message
		ASCENT_INFO("Error: Mesh Blueprint Verify Failed!");
		// show details of what went awry
		verify_info.print();
	}
	else
	{
	//      verify_info.print();
	}

	///////////////////////////////////////////////////////////////////
	// Render with Ascent
	///////////////////////////////////////////////////////////////////

	// add a scene with a pseudocolor plot
	Node scenes;
	scenes["s1/plots/p1/type"] = "pseudocolor";
	scenes["s1/plots/p1/field"] = "temperature";

	//Set the output file name (ascent will add ".png")
	const std::string& png_out = amrex::Concatenate("ascent_render_",n_data_items,5);
	scenes["s1/image_prefix"] = png_out;

	///////////////////////////////////////////////////////////////////

	// setup actions
	Node actions;
	Node &add_act = actions.append();
	add_act["action"] = "add_scenes";
	add_act["scenes"] = scenes;

	// add_act["action"] = "add_extracts";
	// add_act["extracts"] = extracts;

	actions.append()["action"] = "execute";
	actions.append()["action"] = "reset";

	ascent.publish(bp_mesh);

	ascent.execute(actions);

        if(nstep == last_plot) ascent.close();

}
