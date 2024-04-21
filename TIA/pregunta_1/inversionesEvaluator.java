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
        
        
        //mod ejer 1
        boolean invierteEnE21E22 = fenotipo.get(20) > 0 && fenotipo.get(21) > 0;
        boolean invierteEnE23E24 = fenotipo.get(22) > 0 && fenotipo.get(23) > 0;
        
        for (int i = 0; i < fenotipo.size(); i++) {
            if (fenotipo.get(i) > 0) {
            	beneficio += fenotipo.get(i)* Data.beneficioEmpresa[i];
            	riesgo += Data.riesgoEmpresa[i];
            	
            }
        }
        
        //mod de ejer 1
        if (invierteEnE21E22) {
            riesgo *= 1.02;  // Incremento del 2%
        }
        if (invierteEnE23E24) {
            riesgo *= 0.97;  // Decremento del 3%
        }
        
        Objectives objectives = new Objectives();
        objectives.add("Valor del beneficio total - MAX: ", Sign.MAX, beneficio *100); //por 100 porque es en cientos de miles
        objectives.add("Valor del riesgo total - MIN: ", Sign.MIN, riesgo);
        return objectives;
    }
}
