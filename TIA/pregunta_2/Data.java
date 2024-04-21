package ejercicioEXAMEN;

public class Data
{
	public static final double minInversionPorEmpresa = 0.0;
	public static final double maxInversionPorEmpresa = 10.0;
	
	public static final double maxInversionTotal = 100.0;
	
	public static final int numEmpresas = 20;
	
	// beneficio por cada una de las "numEmpresa" empresas 
	public static final double[] beneficioEmpresa =
		{
				0.6, 1.24, 1.2, 2.3, 1.2, 3.4, 3.2, 1.5, 3.2, 2.3, 1.7, 3.6, 1.3, 2.15, 1.4, 2.2, 1.5, 2.9, 2.4, 1.2
		};
	public static final double[] comisiones = {1, 1, 0.5, 2, 1, 1.25, 0.75, 1.2, 1, 2, 1.5, 1.5, 2.1, 1.75, 1.5, 1.75, 1, 1, 0.5, 0.5};
		// riesgo por cada una de las "numEmpresa" empresas 
	public static final int[] riesgoEmpresa =
		{
			1, 3, 2, 3, 1, 7, 6, 2,	5, 5, 2, 6, 3, 7, 6, 3, 2, 4, 5, 5				
		};
}