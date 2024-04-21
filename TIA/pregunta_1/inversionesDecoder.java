package ejercicioEXAMEN;

import java.util.ArrayList;
import org.opt4j.core.genotype.DoubleGenotype;
import org.opt4j.core.problem.Decoder;

public class inversionesDecoder implements Decoder<DoubleGenotype, ArrayList<Double>> {
    @Override
    public ArrayList<Double> decode(DoubleGenotype genotipo) {
        ArrayList<Double> fenotipo = new ArrayList<Double>();
        double inversionTotal = 0.0;
        for (int i = 0; i < genotipo.size(); i++) {
            double inversionEmpresa = genotipo.get(i);
            if (inversionTotal + inversionEmpresa <= Data.maxInversionTotal) {
            	inversionTotal += inversionEmpresa;
            	fenotipo.add(genotipo.get(i));
            } else {
            	fenotipo.add(0.0);
            }
        }

        return fenotipo;
    }
}