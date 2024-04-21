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
        double inversionEnE20 = 0.0;
        double inversionEnE4 = 0.0;
        
        boolean inversionEnE3 = false; //aqui guardo si se invierte o no el 5% en la 3
        
        for (int i = 0; i < fenotipo.size(); i++) {
        	if (fenotipo.get(i) > 0) {
	        	double inversion = fenotipo.get(i);
	            double comision = 0;
	            
	            if(i == 20) {
	            	inversionEnE20 = inversion;
	            }
	            
	            if(i == 4) {
	            	inversionEnE4 = inversion;
	            }
	            
	            if(i == 3 && inversion < 5) {
	            	inversionEnE3 = true; //si invertimos menos del 5% entonces true
	            }
	            // Aplicar comisión si inversion -> + 5%
	            if (i < 20 && inversion > 5) {
	                comision = Data.comisiones[i] * Data.beneficioEmpresa[i] / 100;
	            } //simplemente aqui calculamos lo que hay que restar al beneficio final
	
	            if (inversion > 0) {
	                beneficio += inversion * (Data.beneficioEmpresa[i] - comision); //restamos comision
	                riesgo += Data.riesgoEmpresa[i];
	            }
        	}
        }
        
        if(inversionEnE3) { //hacemos que no se elija esta solucion si no cumple las restricciones del ejercicio
        	beneficio = Double.MIN_VALUE;
            riesgo = Double.MAX_VALUE; 
        }
        
        Objectives objectives = new Objectives();
        objectives.add("Valor del beneficio total - MAX: ", Sign.MAX, beneficio *100); //por 100 porque es en cientos de miles
        objectives.add("Valor del riesgo total - MIN: ", Sign.MIN, riesgo);
        
        objectives.add("Inversion en E20 - MAX", Sign.MAX, inversionEnE20 );
        objectives.add("Inversion en E4 - MIN: ", Sign.MIN, inversionEnE4);
       
        return objectives;
    }
}
