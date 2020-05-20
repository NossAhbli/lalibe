/*!
 *  Functions that perform the contraction of quarks into the two-pion system.
 *  Based on qqbar_w.cc in Chroma
 *  Color is nested inside of spin.
 *  Authors:
 *  Haobo Yan
 *  Andre Walker-Loud
 *  Ben Horz
 */

 // Mathematically, the correlation function is
 //                  _____
 //                  \      -i(px+p'x')                   +     +
 // Cππ(p,p',t)   = >    e           < 0 | π(x)π(x')π(y')π(y) | 0 >
 //                  /
 //                  -----
 //                   x,x'

 // In this program, we implement the calculation with the following formula
 // C(p,p',t) =
 //    tr(Q1(p,y,y) * g5)*tr(P1(p',y',y') * g5)
 //  - tr(Q2(p,y,y') * g5 * P2(p',y',y) * g5)
 //  - tr(Q3(p,y',y) * g5 * P3(p',y,y') * g5)
 //  + tr(Q4(p,y',y') * g5)*tr(P4(p',y,y) * g5)

 // Propagator notations
 // S_u(x, y)  = quark_prop_1
 // S_d(x, y)  = quark_prop_2
 // S_u(x, y') = quark_prop_3
 // S_d(x, y') = quark_prop_4 for pipi scattring

 // S_s(x, y)  = quark_prop_2 for kk scattering
 // S_s(x, y') = quark_prop_4 for kk and kpi scattring

#include "chromabase.h"
#include "pipi_scattering_func_w.h"
#include "../../../chroma/lib/meas/hadron/qqbar_w.cc"

namespace Chroma
{

	void pipi_correlator(CorrelatorType::Correlator& correlator_out, const LatticePropagator& quark_prop_1, const LatticePropagator& quark_prop_2, const LatticePropagator& quark_prop_3, const LatticePropagator& quark_prop_4, const multi2d<int>& origin_list, const int p2max, const int ptot2max, const int t0, const int j_decay)
	{
		int G5 = Ns * Ns - 1;

		// Construct SftMoms
		SftMom phases(p2max, false, j_decay);
		int Nt = phases.numSubsets();

		multi1d<DComplex> tmp_multi1d(Nt); // Use this to give the correlator an initialization
		tmp_multi1d=zero;

		if (origin_list[0] == origin_list[1]) // y=y'
		{
			multi2d<DPropagator> Q(phases.numMom(), Nt);
			Q = zero;

			compute_qqbar(Q, quark_prop_2, quark_prop_1, phases, t0);

			DComplex origin_fix;
			Double origin_phases;
			multi1d<int> mom_comp1, mom_comp2;

			// Loop over all ps, p's and time
			for (int mom_num1 = 0; mom_num1 < phases.numMom(); ++mom_num1)
				for (int mom_num2 = 0; mom_num2 < phases.numMom(); ++mom_num2)
				{
					mom_comp1 = phases.numToMom(mom_num1);
					mom_comp2 = phases.numToMom(mom_num2);

					// Select momenta pairs that lower than the setting p total max
					int ptot2 = 0;
					for (int p_comp = 0; p_comp < 3; ++p_comp)
						ptot2 += (mom_comp1[p_comp] + mom_comp2[p_comp]) * (mom_comp1[p_comp] + mom_comp2[p_comp]);
					if (ptot2 <= ptot2max)
					{
						// Fix the origin
						origin_phases = 0;

						for (int p_comp = 0; p_comp < 3; ++p_comp)
							origin_phases -= (mom_comp1[p_comp] * origin_list[0][p_comp] + mom_comp2[p_comp] * origin_list[1][p_comp]) * 2 * M_PI / Layout::lattSize()[p_comp];

						origin_fix = cmplx(cos(origin_phases), sin(origin_phases));

						correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))] = tmp_multi1d;
						for (int t = 0; t < Nt; ++t)
							correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] = 2 * (trace(Q[mom_num1][t] * Gamma(G5)) * trace(Q[mom_num2][t] * Gamma(G5)) - trace(Q[mom_num1][t] * Gamma(G5) * Q[mom_num2][t] * Gamma(G5))) * origin_fix;
						//correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] =  (trace(Q[mom_num1][t] * Gamma(G5)) * trace(Q[mom_num2][t] * Gamma(G5)))  * origin_fix;
					}
				}

		}
		else // The general case
		{
			multi2d<DPropagator> Q1(phases.numMom(), Nt), P1(phases.numMom(), Nt), Q2(phases.numMom(), Nt), P2(phases.numMom(), Nt);
			Q1 = zero;
			P1 = zero;
			Q2 = zero;
			P2 = zero;

			compute_qqbar(Q1, quark_prop_2, quark_prop_1, phases, t0);
			compute_qqbar(P1, quark_prop_4, quark_prop_3, phases, t0);
			compute_qqbar(Q2, quark_prop_2, quark_prop_3, phases, t0);
			compute_qqbar(P2, quark_prop_4, quark_prop_1, phases, t0);

			//compute_qqbar(Q3, quark_prop_4, quark_prop_1, phases1, t0);
			//compute_qqbar(P3, quark_prop_2, quark_prop_3, phases2, t0);
			//compute_qqbar(Q4, quark_prop_4, quark_prop_3, phases1, t0);
			//compute_qqbar(P4, quark_prop_2, quark_prop_1, phases2, t0);

			// The above Q3 to P4 can be deduced from the first four. So really we only need to compute four of the eight qqbar blocks
			//Q3 = P2;
			//P3 = Q2;
			//Q4 = P1;
			//P4 = Q1;

			DComplex origin_fix;
			Double origin_phases;
			multi1d<int> mom_comp1, mom_comp2;

			for (int mom_num1 = 0; mom_num1 < phases.numMom(); ++mom_num1)
				for (int mom_num2 = 0; mom_num2 < phases.numMom(); ++mom_num2)
				{
					mom_comp1 = phases.numToMom(mom_num1);
					mom_comp2 = phases.numToMom(mom_num2);

					// Select momenta pairs that lower than the setting p total max
					int ptot2 = 0;
					for (int p_comp = 0; p_comp < 3; ++p_comp)
						ptot2 += (mom_comp1[p_comp] + mom_comp2[p_comp]) * (mom_comp1[p_comp] + mom_comp2[p_comp]);
					if (ptot2 <= ptot2max)
					{
						// Fix the origin
						origin_phases = 0;

						for (int p_comp = 0; p_comp < 3; ++p_comp)
						//origin_phases -= (mom_comp1[p_comp] * origin_list[0][p_comp] + mom_comp2[p_comp] * origin_list[1][p_comp]) * 2 * M_PI / Layout::lattSize()[p_comp];
						origin_phases -= ((mom_comp2[p_comp]) *( origin_list[0][p_comp]-origin_list[1][p_comp])) * 2 * M_PI / Layout::lattSize()[p_comp];

						origin_fix = cmplx(cos(origin_phases), sin(origin_phases));

						correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))] = tmp_multi1d;
						for (int t = 0; t < Nt; ++t)
							correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] = (trace(Q1[mom_num1][t] * Gamma(G5)) * trace(P1[mom_num2][t] * Gamma(G5)) - trace(Q2[mom_num1][t] * Gamma(G5) * P2[mom_num2][t] * Gamma(G5)) - trace(P2[mom_num1][t] * Gamma(G5) * Q2[mom_num2][t] * Gamma(G5)) + trace(P1[mom_num1][t] * Gamma(G5)) * trace(Q1[mom_num2][t] * Gamma(G5))) * origin_fix;
						//correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] = (trace(Q1[mom_num1][t] * Gamma(G5)) * trace(P1[mom_num2][t] * Gamma(G5)) ) * origin_fix;
					}
				}

		}

	}

	void kpi_correlator(CorrelatorType::Correlator& correlator_out, const LatticePropagator& quark_prop_1, const LatticePropagator& quark_prop_2, const LatticePropagator& quark_prop_3, const LatticePropagator& quark_prop_4, const multi2d<int>& origin_list, const int p2max, const int ptot2max, const int t0, const int j_decay)
	{
		int G5 = Ns * Ns - 1;

		// Construct SftMoms
		SftMom phases(p2max, false, j_decay);
		int Nt = phases.numSubsets();

		multi1d<DComplex> tmp_multi1d(Nt); // Use this to give the correlator an initialization
		tmp_multi1d=zero;

		multi2d<DPropagator> Q1(phases.numMom(), Nt), P1(phases.numMom(), Nt), Q2(phases.numMom(), Nt), P2(phases.numMom(), Nt);
		Q1 = zero;
		P1 = zero;
		Q2 = zero;
		P2 = zero;

		compute_qqbar(Q1, quark_prop_2, quark_prop_1, phases, t0);
		compute_qqbar(P1, quark_prop_4, quark_prop_3, phases, t0);
		compute_qqbar(Q2, quark_prop_2, quark_prop_3, phases, t0);
		compute_qqbar(P2, quark_prop_4, quark_prop_1, phases, t0);

		DComplex origin_fix;
		Double origin_phases;
		multi1d<int> mom_comp1, mom_comp2;

		for (int mom_num1 = 0; mom_num1 < phases.numMom(); ++mom_num1)
			for (int mom_num2 = 0; mom_num2 < phases.numMom(); ++mom_num2)
			{
				mom_comp1 = phases.numToMom(mom_num1);
				mom_comp2 = phases.numToMom(mom_num2);

				// Select momenta pairs that lower than the setting p total max
				int ptot2 = 0;
				for (int p_comp = 0; p_comp < 3; ++p_comp)
					ptot2 += (mom_comp1[p_comp] + mom_comp2[p_comp]) * (mom_comp1[p_comp] + mom_comp2[p_comp]);
				if (ptot2 <= ptot2max)
				{
					// Fix the origin
					origin_phases = 0;

//					for (int p_comp = 0; p_comp < 3; ++p_comp)
	//					origin_phases -= (mom_comp1[p_comp] * origin_list[0][p_comp] + mom_comp2[p_comp] * origin_list[1][p_comp]) * 2 * M_PI / Layout::lattSize()[p_comp];
																		for (int p_comp = 0; p_comp < 3; ++p_comp)
																		{
																			//origin_phases -= (mom_comp1[p_comp] * origin_list[0][p_comp] + mom_comp2[p_comp] * origin_list[1][p_comp]) * 2 * M_PI / Layout::lattSize()[p_comp];
																			origin_phases -= (mom_comp1[p_comp]+mom_comp2[p_comp]) *( origin_list[0][p_comp]) * 2 * M_PI / Layout::lattSize()[p_comp];

																		}
					origin_fix = cmplx(cos(origin_phases), sin(origin_phases));

					correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))] = tmp_multi1d;
					for (int t = 0; t < Nt; ++t)
						correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] = (trace(Q1[mom_num1][t] * Gamma(G5)) * trace(P1[mom_num2][t] * Gamma(G5)) - trace(Q2[mom_num1][t] * Gamma(G5) * P2[mom_num2][t] * Gamma(G5))) * origin_fix;
				}
			}

	}









		void pipi_correlator_debug(CorrelatorType::Correlator& correlator_out, const LatticePropagator& quark_prop_1, const LatticePropagator& quark_prop_2, const LatticePropagator& quark_prop_3, const LatticePropagator& quark_prop_4, const multi2d<int>& origin_list, const int p2max, const int ptot2max, const int t0, const int j_decay, const int diagram)
		{
			int G5 = Ns * Ns - 1;
//QDPIO::cout<<"origin list: ["<<origin_list[0][0]<<origin_list[0][1]<<origin_list[0][2]<<origin_list[0][3]<<origin_list[1][0]<<origin_list[1][1]<<origin_list[1][2]<<origin_list[1][3]<<"]"<<std::endl;
			// Construct SftMoms
			SftMom phases(p2max, false, j_decay);
			int Nt = phases.numSubsets();

			// Judge if the origin is the same
			bool same_origin = true;
			for (int o_comp = 0; o_comp < origin_list.size2(); ++o_comp)
				if (origin_list[0][o_comp] != origin_list[1][o_comp])
				{
					same_origin = false;
					break;
				}
			multi1d<DComplex> tmp_multi1d(Nt); // Use this to give correlator a Initialization
			tmp_multi1d=zero;
	QDPIO::cout<<(origin_list[0] == origin_list[1])<<std::endl;
			if (origin_list[0] == origin_list[1]) // y=y'
			{
				multi2d<DPropagator> Q(phases.numMom(), Nt);
				Q = zero;

				compute_qqbar(Q, quark_prop_2, quark_prop_1, phases, t0);

				DComplex origin_fix;
				Double origin_phases;
				multi1d<int> mom_comp1, mom_comp2;

				// Loop over all ps, p's and time
				for (int mom_num1 = 0; mom_num1 < phases.numMom(); ++mom_num1)
					for (int mom_num2 = 0; mom_num2 < phases.numMom(); ++mom_num2)
					{
						mom_comp1 = phases.numToMom(mom_num1);
						mom_comp2 = phases.numToMom(mom_num2);

						// Select momenta pairs that lower than the setting p total max
						int ptot2 = 0;
						for (int p_comp = 0; p_comp < 3; ++p_comp)
							ptot2 += (mom_comp1[p_comp] + mom_comp2[p_comp]) * (mom_comp1[p_comp] + mom_comp2[p_comp]);
						if (ptot2 <= ptot2max)
						{
							// Fix the origin
							origin_phases = 0;

							for (int p_comp = 0; p_comp < 3; ++p_comp)
								origin_phases -= (mom_comp1[p_comp] * origin_list[0][p_comp] + mom_comp2[p_comp] * origin_list[1][p_comp]) * 2 * M_PI / Layout::lattSize()[p_comp];



							origin_fix = cmplx(cos(origin_phases), sin(origin_phases));

							correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))] = tmp_multi1d;
							if (diagram == 1 || diagram == 4)
							{
								for (int t = 0; t < Nt; ++t)
									correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] =  (trace(Q[mom_num1][t] * Gamma(G5)) * trace(Q[mom_num2][t] * Gamma(G5)))  * origin_fix;
							}
							else if (diagram == 2 || diagram == 3)
							{
								for (int t = 0; t < Nt; ++t)
									correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] =  (- trace(Q[mom_num1][t] * Gamma(G5) * Q[mom_num2][t] * Gamma(G5)))  * origin_fix;
							}
							else
							{
								for (int t = 0; t < Nt; ++t)
									correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] = 2 * (trace(Q[mom_num1][t] * Gamma(G5)) * trace(Q[mom_num2][t] * Gamma(G5)) - trace(Q[mom_num1][t] * Gamma(G5) * Q[mom_num2][t] * Gamma(G5))) * origin_fix;
							}
						}
					}

			}
			else // The general case
			{
				multi2d<DPropagator> Q1(phases.numMom(), Nt), P1(phases.numMom(), Nt), Q2(phases.numMom(), Nt), P2(phases.numMom(), Nt);
				Q1 = zero;
				P1 = zero;
				Q2 = zero;
				P2 = zero;

				compute_qqbar(Q1, quark_prop_2, quark_prop_1, phases, t0);
				compute_qqbar(P1, quark_prop_4, quark_prop_3, phases, t0);
				compute_qqbar(Q2, quark_prop_2, quark_prop_3, phases, t0);
				compute_qqbar(P2, quark_prop_4, quark_prop_1, phases, t0);

				//compute_qqbar(Q3, quark_prop_4, quark_prop_1, phases1, t0);
				//compute_qqbar(P3, quark_prop_2, quark_prop_3, phases2, t0);
				//compute_qqbar(Q4, quark_prop_4, quark_prop_3, phases1, t0);
				//compute_qqbar(P4, quark_prop_2, quark_prop_1, phases2, t0);

				// The above Q3 to P4 can be deduced from the first four. So really we only need to compute four of the eight qqbar blocks
				//Q3 = P2;
				//P3 = Q2;
				//Q4 = P1;
				//P4 = Q1;

				DComplex origin_fix1, origin_fix2;
				Double origin_phases1, origin_phases2;
				multi1d<int> mom_comp1, mom_comp2;

				for (int mom_num1 = 0; mom_num1 < phases.numMom(); ++mom_num1)
					for (int mom_num2 = 0; mom_num2 < phases.numMom(); ++mom_num2)
					{
						mom_comp1 = phases.numToMom(mom_num1);
						mom_comp2 = phases.numToMom(mom_num2);

						// Select momenta pairs that lower than the setting p total max
						int ptot2 = 0;
						for (int p_comp = 0; p_comp < 3; ++p_comp)
							ptot2 += (mom_comp1[p_comp] + mom_comp2[p_comp]) * (mom_comp1[p_comp] + mom_comp2[p_comp]);
						if (ptot2 <= ptot2max)
						{
							// Fix the origin
							origin_phases1 = 0;
							origin_phases2 = 0;

							//for (int p_comp = 0; p_comp < 3; ++p_comp)
							//	origin_phases -= (mom_comp1[p_comp] * origin_list[0][p_comp] + mom_comp2[p_comp] * origin_list[1][p_comp]) * 2 * M_PI / Layout::lattSize()[p_comp];

																			for (int p_comp = 0; p_comp < 3; ++p_comp)
																			{
																				//origin_phases -= (mom_comp1[p_comp] * origin_list[0][p_comp] + mom_comp2[p_comp] * origin_list[1][p_comp]) * 2 * M_PI / Layout::lattSize()[p_comp];
																				origin_phases1 -= (mom_comp1[p_comp]+mom_comp2[p_comp]) *( origin_list[0][p_comp]) * 2 * M_PI / Layout::lattSize()[p_comp];
																				origin_phases2 -= (mom_comp1[p_comp]+mom_comp2[p_comp]) *( origin_list[1][p_comp]) * 2 * M_PI / Layout::lattSize()[p_comp];

																			}
																			QDPIO::cout<<"["<<mom_comp1[0]<<mom_comp1[1]<<mom_comp1[2]<<","<<mom_comp2[0]<<mom_comp2[1]<<mom_comp2[2]<<"]"<<origin_phases1<<std::endl;



																			origin_fix1 = cmplx(cos(origin_phases1), sin(origin_phases1));
																			origin_fix2 = cmplx(cos(origin_phases2), sin(origin_phases2));

							correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))] = tmp_multi1d;
							if (diagram == 1)
							{
								for (int t = 0; t < Nt; ++t)
									correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] = trace(Q1[mom_num1][t] * Gamma(G5)) * trace(P1[mom_num2][t] * Gamma(G5)) * origin_fix1;
							}
							else if (diagram == 2)
							{
								for (int t = 0; t < Nt; ++t)
									correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] =  - trace(Q2[mom_num1][t] * Gamma(G5) * P2[mom_num2][t] * Gamma(G5))  * origin_fix1;
							}
							else if (diagram == 3)
							{
								for (int t = 0; t < Nt; ++t)
									correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] =  - trace(P2[mom_num1][t] * Gamma(G5) * Q2[mom_num2][t] * Gamma(G5))  * origin_fix2;
							}
							else if (diagram == 4)
							{
								for (int t = 0; t < Nt; ++t)
									correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] = trace(P1[mom_num1][t] * Gamma(G5)) * trace(Q1[mom_num2][t] * Gamma(G5)) * origin_fix2;
							}
							else
							{
								for (int t = 0; t < Nt; ++t)
									correlator_out[std::make_pair(std::make_tuple(mom_comp1[0], mom_comp1[1], mom_comp1[2]), std::make_tuple(mom_comp2[0], mom_comp2[1], mom_comp2[2]))][t] = (trace(Q1[mom_num1][t] * Gamma(G5)) * trace(P1[mom_num2][t] * Gamma(G5)) - trace(Q2[mom_num1][t] * Gamma(G5) * P2[mom_num2][t] * Gamma(G5))) * origin_fix1 + (- trace(P2[mom_num1][t] * Gamma(G5) * Q2[mom_num2][t] * Gamma(G5)) + trace(P1[mom_num1][t] * Gamma(G5)) * trace(Q1[mom_num2][t] * Gamma(G5))) * origin_fix2;
							}
						}
					}

			}

		}



} // End namespace Chroma
