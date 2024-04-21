package ejercicioEXAMEN;

import java.util.Random;
import org.opt4j.core.genotype.DoubleGenotype;
import org.opt4j.core.problem.Creator;

public class inversionesCreator implements Creator<DoubleGenotype>{

	public DoubleGenotype create() {
		DoubleGenotype genotipo = new DoubleGenotype(Data.minInversionPorEmpresa, Data.maxInversionPorEmpresa);
		genotipo.init(new Random(), Data.numEmpresas);
		return genotipo;
	}
}
