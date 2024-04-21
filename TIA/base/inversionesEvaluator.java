package ejercicioEXAMEN;

import java.util.ArrayList;
import org.opt4j.core.Objective.Sign;
import org.opt4j.core.Objectives;
import org.opt4j.core.problem.Evaluator;

public class inversionesEvaluator implements Evaluator<ArrayList<Double>> {
    @Override
    public Objectives evaluate(ArrayList<Double> fenotipo) {
        

        double beneficio = 0.0;
        double riesgo = 0.0;
        //double inversionTotal = 0.0; 

        for (int i = 0; i < fenotipo.size(); i++) {
            if (fenotipo.get(i) > 0) {
            	beneficio += fenotipo.get(i)* Data.beneficioEmpresa[i];
            	riesgo += Data.riesgoEmpresa[i];
            	//inversionTotal += fenotipo.get(i);
            }
        }
        // Verifica si la la inversion total no pasa de 100
        //if (inversionTotal > Data.maxInversionTotal) {
        //	beneficio = Double.MIN_VALUE;
        //	riesgo = Double.MAX_VALUE; //podrias quitarla
        //}
        
        Objectives objectives = new Objectives();
        objectives.add("Valor del beneficio total - MAX: ", Sign.MAX, beneficio *100); //por 100 porque es en cientos de miles
        objectives.add("Valor del riesgo total - MIN: ", Sign.MIN, riesgo);
        return objectives;
    }
}
