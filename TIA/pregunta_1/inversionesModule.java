package ejercicioEXAMEN;
import org.opt4j.core.problem.ProblemModule;

public class inversionesModule extends ProblemModule {
	protected void config() {
		bindProblem(inversionesCreator.class, inversionesDecoder.class, inversionesEvaluator.class);
	}
}
