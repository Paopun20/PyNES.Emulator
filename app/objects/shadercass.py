from __future__ import annotations
import re
import textwrap
from typing import Type, Final, Optional, List, Dict, Set
from enum import Enum
from dataclasses import dataclass, field
from string import Template

__doc_template__ = Template("""
${name}${shader_type}

# Description
${description}

# Uniforms
${uniforms}

# Attributes
${attributes}

# Varyings
${varyings}

# Functions
${functions}
""")


class ShaderType(Enum):
    """Type of shader program"""
    VERTEX = "vertex"
    FRAGMENT = "fragment"
    GEOMETRY = "geometry"
    COMPUTE = "compute"
    UNKNOWN = "unknown"


class ShaderUniformEnum(Enum):
    """GLSL uniform types"""
    FLOAT = "float"
    INT = "int"
    UINT = "uint"
    BOOL = "bool"
    VEC2 = "vec2"
    VEC3 = "vec3"
    VEC4 = "vec4"
    IVEC2 = "ivec2"
    IVEC3 = "ivec3"
    IVEC4 = "ivec4"
    UVEC2 = "uvec2"
    UVEC3 = "uvec3"
    UVEC4 = "uvec4"
    BVEC2 = "bvec2"
    BVEC3 = "bvec3"
    BVEC4 = "bvec4"
    MAT2 = "mat2"
    MAT3 = "mat3"
    MAT4 = "mat4"
    MAT2X3 = "mat2x3"
    MAT2X4 = "mat2x4"
    MAT3X2 = "mat3x2"
    MAT3X4 = "mat3x4"
    MAT4X2 = "mat4x2"
    MAT4X3 = "mat4x3"
    SAMPLER1D = "sampler1D"
    SAMPLER2D = "sampler2D"
    SAMPLER3D = "sampler3D"
    SAMPLERCUBE = "samplerCube"
    SAMPLER2DSHADOW = "sampler2DShadow"


class ShaderQualifier(Enum):
    """GLSL variable qualifiers"""
    UNIFORM = "uniform"
    ATTRIBUTE = "attribute"
    VARYING = "varying"
    IN = "in"
    OUT = "out"
    INOUT = "inout"


@dataclass
class ShaderVariable:
    """Represents a shader variable (uniform, attribute, varying)"""
    name: str
    type: ShaderUniformEnum
    qualifier: ShaderQualifier
    array_size: Optional[int] = None
    default_value: Optional[str] = None
    line_number: int = 0

    def __str__(self) -> str:
        array_part = f"[{self.array_size}]" if self.array_size else ""
        return f"{self.qualifier.value} {self.type.value} {self.name}{array_part};"

    @property
    def is_array(self) -> bool:
        return self.array_size is not None


@dataclass
class ShaderFunction:
    """Represents a function defined in shader code"""
    name: str
    return_type: str
    parameters: List[str] = field(default_factory=list)
    line_number: int = 0

    def __str__(self) -> str:
        params = ", ".join(self.parameters) if self.parameters else ""
        return f"{self.return_type} {self.name}({params})"


@dataclass
class ShaderValidationError:
    """Represents a validation error in shader code"""
    line: int
    message: str
    severity: str = "error"  # error, warning, info

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] Line {self.line}: {self.message}"


class ShaderValidator:
    """Validates GLSL shader code"""
    
    @staticmethod
    def validate(code: str, shader_type: ShaderType) -> List[ShaderValidationError]:
        errors = []
        lines = code.split('\n')
        
        # Check for main function
        if not re.search(r'\bvoid\s+main\s*\(', code):
            errors.append(ShaderValidationError(
                line=0,
                message="Missing main() function",
                severity="error"
            ))
        
        # Check for balanced braces
        brace_count = code.count('{') - code.count('}')
        if brace_count != 0:
            errors.append(ShaderValidationError(
                line=0,
                message=f"Unbalanced braces (difference: {brace_count})",
                severity="error"
            ))
        
        # Check for balanced parentheses
        paren_count = code.count('(') - code.count(')')
        if paren_count != 0:
            errors.append(ShaderValidationError(
                line=0,
                message=f"Unbalanced parentheses (difference: {paren_count})",
                severity="error"
            ))
        
        # Check for deprecated functions/variables
        deprecated_gl = {
            'gl_FragColor': 'Use out variables instead',
            'gl_FragData': 'Use out variables instead',
            'texture2D': 'Use texture() instead',
            'texture3D': 'Use texture() instead',
            'textureCube': 'Use texture() instead',
        }
        
        for i, line in enumerate(lines, 1):
            for deprecated, suggestion in deprecated_gl.items():
                if deprecated in line and not line.strip().startswith('//'):
                    errors.append(ShaderValidationError(
                        line=i,
                        message=f"Deprecated: '{deprecated}'. {suggestion}",
                        severity="warning"
                    ))
        
        # Check for common mistakes
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped and not stripped.startswith('//'):
                # Check for assignment in conditionals
                if re.search(r'if\s*\([^)]*=[^=]', stripped):
                    errors.append(ShaderValidationError(
                        line=i,
                        message="Possible assignment instead of comparison in conditional",
                        severity="warning"
                    ))
                
                # Check for semicolon after control structures
                if re.search(r'(if|for|while)\s*\([^)]*\)\s*;', stripped):
                    errors.append(ShaderValidationError(
                        line=i,
                        message="Semicolon after control structure",
                        severity="warning"
                    ))
        
        return errors


class Shader:
    """
    Wraps a Python class docstring as shader code. Usable as a decorator.
    
    Enhanced features:
    - Automatic shader type detection
    - Comprehensive uniform, attribute, and varying parsing
    - Function signature extraction
    - GLSL validation with error reporting
    - Preprocessor directive handling
    - Better documentation generation
    """

    def __init__(
        self,
        description: Optional[str] = "",
        shader_type: Optional[ShaderType] = None,
        validate: bool = True,
        strip_comments: bool = True
    ):
        self._description: Final[str] = description or ""
        self._shader_type: Optional[ShaderType] = shader_type
        self._validate_code: Final[bool] = validate
        self._strip_comments: Final[bool] = strip_comments
        
        self._name: str = ""
        self._code: str = ""
        self._original_code: str = ""
        self._uniforms: List[ShaderVariable] = []
        self._attributes: List[ShaderVariable] = []
        self._varyings: List[ShaderVariable] = []
        self._functions: List[ShaderFunction] = []
        self._preprocessor_defines: Dict[str, str] = {}
        self._validation_errors: List[ShaderValidationError] = []
        self._cls: Optional[Type] = None
        self._doc: str = ""
        self.__doc__: str = ""

    def __call__(self, cls: Type) -> Shader:
        if not cls.__doc__:
            raise ValueError("Shader class must have a docstring containing shader code")

        self._name = cls.__name__
        self._original_code = textwrap.dedent(cls.__doc__).strip()
        self._cls = cls
        
        # Parse preprocessor directives
        self._preprocessor_defines = self._parse_defines(self._original_code)
        
        # Clean code
        code = self._original_code
        if self._strip_comments:
            code = self._remove_comments(code)
        code = self._normalize_whitespace(code)
        self._code = code
        
        # Detect shader type if not specified
        if self._shader_type is None:
            self._shader_type = self._detect_shader_type(code)
        
        # Parse shader components
        self._uniforms = self._parse_variables(code, ShaderQualifier.UNIFORM)
        self._attributes = self._parse_variables(code, ShaderQualifier.ATTRIBUTE)
        self._varyings = self._parse_variables(code, ShaderQualifier.VARYING)
        self._functions = self._parse_functions(code)
        
        # Validate if requested
        if self._validate_code:
            self._validation_errors = ShaderValidator.validate(code, self._shader_type)
        
        # Generate documentation
        self._doc = self._generate_documentation()
        self.__doc__ = self._doc

        return self

    def _remove_comments(self, code: str) -> str:
        """Remove C-style comments from code"""
        # Remove multi-line comments
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        # Remove single-line comments
        code = re.sub(r"//.*", "", code)
        return code

    def _normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace in code"""
        # Remove excessive blank lines
        code = re.sub(r"\n\s*\n\s*\n+", "\n\n", code)
        return code.strip()

    def _parse_defines(self, code: str) -> Dict[str, str]:
        """Parse #define preprocessor directives"""
        defines = {}
        pattern = re.compile(r"#define\s+(\w+)(?:\s+(.+?))?(?:\n|$)")
        for match in pattern.finditer(code):
            name, value = match.groups()
            defines[name] = value.strip() if value else ""
        return defines

    def _detect_shader_type(self, code: str) -> ShaderType:
        """Detect shader type from code content"""
        code_lower = code.lower()
        
        # Check for shader-specific outputs/variables
        if 'gl_position' in code_lower:
            return ShaderType.VERTEX
        if 'gl_fragcolor' in code_lower or 'gl_fragdata' in code_lower:
            return ShaderType.FRAGMENT
        if 'gl_in' in code_lower and 'emitvertex' in code_lower:
            return ShaderType.GEOMETRY
        
        # Check for shader-specific keywords
        if re.search(r'\blayout\s*\(.*local_size', code_lower):
            return ShaderType.COMPUTE
        
        return ShaderType.UNKNOWN

    def _parse_variables(
        self,
        code: str,
        qualifier: ShaderQualifier
    ) -> List[ShaderVariable]:
        """Parse shader variables with given qualifier"""
        variables: List[ShaderVariable] = []
        
        # Pattern matches: qualifier type name[array_size]; or qualifier type name;
        pattern = re.compile(
            rf"\b{qualifier.value}\s+(\w+)\s+(\w+)(?:\[(\d+)\])?\s*;"
        )
        
        for i, line in enumerate(code.split('\n'), 1):
            for match in pattern.finditer(line):
                type_str, name, array_size = match.groups()
                try:
                    type_enum = ShaderUniformEnum(type_str)
                except ValueError:
                    continue
                
                variables.append(ShaderVariable(
                    name=name,
                    type=type_enum,
                    qualifier=qualifier,
                    array_size=int(array_size) if array_size else None,
                    line_number=i
                ))
        
        return variables

    def _parse_functions(self, code: str) -> List[ShaderFunction]:
        """Parse function signatures from shader code"""
        functions: List[ShaderFunction] = []
        
        # Pattern matches function declarations
        pattern = re.compile(
            r"\b(\w+)\s+(\w+)\s*\(([^)]*)\)\s*(?:\{|;)"
        )
        
        for i, line in enumerate(code.split('\n'), 1):
            for match in pattern.finditer(line):
                return_type, name, params_str = match.groups()
                
                # Skip reserved keywords
                if name in ('if', 'for', 'while', 'switch'):
                    continue
                
                # Parse parameters
                params = []
                if params_str.strip():
                    for param in params_str.split(','):
                        param = param.strip()
                        if param:
                            params.append(param)
                
                functions.append(ShaderFunction(
                    name=name,
                    return_type=return_type,
                    parameters=params,
                    line_number=i
                ))
        
        return functions

    def _generate_documentation(self) -> str:
        """Generate comprehensive documentation"""
        shader_type_str = f" ({self._shader_type.value})" if self._shader_type != ShaderType.UNKNOWN else ""
        
        uniform_list = "\n".join(str(u) for u in self._uniforms) or "(none)"
        attribute_list = "\n".join(str(a) for a in self._attributes) or "(none)"
        varying_list = "\n".join(str(v) for v in self._varyings) or "(none)"
        function_list = "\n".join(str(f) for f in self._functions if f.name != 'main') or "(none)"
        
        doc = __doc_template__.substitute(
            name=self._name,
            shader_type=shader_type_str,
            description=self._description or "(no description)",
            uniforms=uniform_list,
            attributes=attribute_list,
            varyings=varying_list,
            functions=function_list
        ).strip()
        
        # Add validation errors if any
        if self._validation_errors:
            doc += "\n\n# Validation Issues\n"
            doc += "\n".join(str(e) for e in self._validation_errors)
        
        return doc

    @property
    def name(self) -> str:
        return self._name

    @property
    def code(self) -> str:
        return self._code

    @property
    def original_code(self) -> str:
        return self._original_code

    @property
    def shader_type(self) -> ShaderType:
        return self._shader_type

    @property
    def uniforms(self) -> List[ShaderVariable]:
        return self._uniforms

    @property
    def attributes(self) -> List[ShaderVariable]:
        return self._attributes

    @property
    def varyings(self) -> List[ShaderVariable]:
        return self._varyings

    @property
    def functions(self) -> List[ShaderFunction]:
        return self._functions

    @property
    def defines(self) -> Dict[str, str]:
        return self._preprocessor_defines

    @property
    def validation_errors(self) -> List[ShaderValidationError]:
        return self._validation_errors

    @property
    def is_valid(self) -> bool:
        """Check if shader has no validation errors"""
        return not any(e.severity == "error" for e in self._validation_errors)

    def get_uniform_names(self) -> Set[str]:
        """Get set of all uniform names"""
        return {u.name for u in self._uniforms}

    def get_uniform_by_name(self, name: str) -> Optional[ShaderVariable]:
        """Get uniform by name"""
        for uniform in self._uniforms:
            if uniform.name == name:
                return uniform
        return None

    def __str__(self) -> str:
        type_str = f" [{self._shader_type.value}]" if self._shader_type != ShaderType.UNKNOWN else ""
        return f"<Shader {self._name}{type_str} ({len(self._code.splitlines())} lines)>"

    def __repr__(self) -> str:
        return self.__str__()