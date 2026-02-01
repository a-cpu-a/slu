// ============================================================================
// BASE CLASSES
// ============================================================================

/// Use this when you dont need a preSpace
class CompoundNode {
    constructor(type) {
        this.type = type;
    }
}
class Node extends CompoundNode {
    constructor(type) {
        super(type)
        this.preSpace = "";
    }
}

class Token extends Node {
    constructor(txt) {
        super("Token");
        this.txt = txt;
    }
}

class Safety extends Node {
    // kind: "default" | "unsafe" | "safe"
    constructor() {
        super("Safety");
        this.kind = "default";
    }
}

class Export extends Node {
    // kind: "none" | "full"
    constructor() {
        super("Export");
        this.kind = "none";
    }
}

class Name extends Node {
    constructor() {
        super("Name");
        this.name = "";
    }
}
class Num extends Node {
    constructor(num) {
        super("Num");
        this.num = num; // Including 0x ... p+...
    }
}
class Str extends Node {
    constructor(str) {
        super("Str");
        this.str = str; // Including the quotes or [[]]
    }
}
class TuplableName extends CompoundNode {
    constructor(type) { super(type); }
}
class NameTuplableName extends TuplableName {
    constructor() {
        super("NameTuplableName");
        this.name = new Name();
    }
}
class NumTuplableName extends TuplableName {
    constructor() {
        super("NumTuplableName");
        this.name = new Num();
    }
}

// ============================================================================
// HELPERS
// ============================================================================

class DelimitedList extends CompoundNode {
    constructor(type) {
        super("DelimitedList");
        this.itemType = type;
        this.items = []; // Array of {v: type, sep: OptToken(...)}
    }
}

class OptToken extends Node {
    constructor(txt) {
        super("OptToken");
        this.txt = txt;
        this.present = false; // true if parsed
    }
}

class OptBlockNode extends CompoundNode {
    constructor() {
        super("OptBlockNode");
        this.present = false; // true if block exists
        this.block = new BlockNode();
    }
}

// ============================================================================
// PATTERNS (sPat, dPat, uncondDestrPat)
// ============================================================================

// destrSpec ::= sPat | ["mut"] {typePreop}
class DestrSpec extends CompoundNode {
    constructor(type) { super(type); }
}
class SimplePatDestrSpec extends DestrSpec {
    constructor() {
        super("SimplePatDestrSpec");
        this.type = new SimplePat();
    }
}
class OpDestrSpec extends DestrSpec {
    constructor() {
        super("OpDestrSpec");
        this.mutKw = new OptToken("mut");
        this.ops = []; // Array of Token|UnOp
    }
}
class Pat extends CompoundNode {
    constructor(type) { super(type); }
}
class SimplePat extends Pat {
    constructor() {
        super("SimplePat");
        this.expression = Expr();
    }
}
class DestrPat extends UncondDestrPat {
    constructor(type) { super(type); }
}
class VarDestrPat extends DestrPat {
    constructor() {
        super("VarDestrPat");
        this.base = new UncondVarDestrPat();
        this.eq = new OptToken("=");
        this.valPat = new SimplePat(); // $<Needs eq>
    }
}
class PatFieldDestrPat extends DestrPat {
    constructor() {
        super("PatFieldDestrPat");
        this.specifiers = new DestrSpec();
        this.openBrace = new Token("{");
        this.fields = new DelimitedList("Pat");
        this.extraFields = new OptToken("..");
        this.closeBrace = new Token("}");
    }
}
class FieldDestrPat extends DestrPat {
    constructor() {
        super("FieldDestrPat");
        this.specifiers = new DestrSpec();
        this.openBrace = new Token("{");
        this.fields = new DelimitedList("FieldDestrField");
        this.extraFields = new OptToken("..");
        this.closeBrace = new Token("}");
    }
}
class FieldDestrField extends CompoundNode {
    constructor() {
        super("FieldDestrField");
        this.openPipe = new Token("|");
        this.var = new TuplableName();
        this.closePipe = new Token("|");
        this.pat = new Pat();
    }
}
class UncondDestrPat extends Pat {
    constructor(type) { super(type); }
}
class AlwaysDestrPat extends UncondDestrPat {
    constructor() {
        super("AlwaysDestrPat");
        this.us = new Token("_");
    }
}
class UncondVarDestrPat extends UncondDestrPat {
    constructor() {
        super("UncondVarDestrPat");
        this.specifiers = new DestrSpec();
        this.name = new Name();
    }
}
class UncondPatFieldDestrPat extends UncondDestrPat {
    constructor() {
        super("UncondPatFieldDestrPat");
        this.specifiers = new DestrSpec();
        this.openBrace = new Token("{");
        this.fields = new DelimitedList("UncondFieldDestrField");
        this.extraFields = new OptToken("..");
        this.closeBrace = new Token("}");
    }
}
class UncondFieldDestrPat extends UncondDestrPat {
    constructor() {
        super("UncondFieldDestrPat");
        this.specifiers = new DestrSpec();
        this.openBrace = new Token("{");
        this.fields = new DelimitedList("UncondPatFieldDestrPat");
        this.extraFields = new OptToken("..");
        this.closeBrace = new Token("}");
    }
}
class UncondFieldDestrField extends CompoundNode {
    constructor() {
        super("UncondFieldDestrField");
        this.openPipe = new Token("|");
        this.var = new TuplableName();
        this.closePipe = new Token("|");
        this.pat = new UncondDestrPat();
    }
}
// ============================================================================
// EXPRESSIONS & STATEMENTS
// ============================================================================

class Stat extends CompoundNode {
    constructor(type) { super(type); }
}

class Expr extends CompoundNode {
    constructor() {
        super("Expr");
    }
}
class UnOp extends CompoundNode {
    constructor() {
        super("UnOp");
    }
}
class Args extends CompoundNode {
    constructor(type) { super(type); }
}
class TableArgs extends Args {
    constructor() {
        super("TableArgs");
        this.args = new TableConstructor();
    }
}
class ParenArgs extends Args {
    constructor() {
        super("ParenArgs");
        this.openParen = new Token("(");
        this.args = new DelimitedList("Expr");
        this.closeParen = new Token(")");
    }
}
class NumArgs extends Args {
    constructor() {
        super("NumArgs");
        this.args = new Num();
    }
}
class StrArgs extends Args {
    constructor() {
        super("StrArgs");
        this.args = new Str();
    }
}

class BlockNode extends Node {
    constructor() {
        super("BlockNode");
        this.openBrace = new Token("{");
        this.stats = [];
        this.retStat = new Stat();
        this.closeBrace = new Token("}");
    }
}

class TableConstructor extends Expr {
    constructor() {
        super("TableConstructor");
        this.openBrace = new Token("{");
        this.fields = new DelimitedList("Field");
        this.closeBrace = new Token("}");
    }
}

class ModPath extends Node {
    constructor() {
        super("ModPath");
        this.root = new Token(); // Name | Token("self") | Token("crate") | Token(":>")
        this.segments = []; // { dc: Token("::"), name: Name }
    }
}
class MatchItem extends Node {
    constructor() {
        super("MatchItem");
        this.pat = new Pat();

        this.ifKw = new OptToken("if");
        this.ifExpr = new Expr(); // $<Needs ifKw>

        this.arrow = new Token("=>");
        this.expr = new Expr();
    }
}
class MatchTypeBlock extends Node {
    constructor() {
        super("MatchTypeBlock");

        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>

        this.openBrace = new Token("{");
        this.items = new DelimitedList("MatchItem");
        this.closeBrace = new Token("}");
    }
}

class TypedParam extends Node {
    constructor() {
        super("TypedParam");
        this.constKw = new OptToken("const");
        this.name = new Name();
        this.eq = new Token("=");
        this.type = new Expr();
    }
}

// ============================================================================
// GLOBAL DECLARATIONS (globstat)
// ============================================================================

class GlobStat extends Node {
    constructor(type) { super(type); }
}

// optexport "struct" Name ["(" [params] ")"] tableconstructor
class StructDecl extends GlobStat {
    constructor() {
        super("StructDecl");
        this.export = new Export();
        this.structKw = new Token("struct");
        this.name = new Name();

        this.openParen = new OptToken("(");
        this.params = new DelimitedList("TypedParam"); // $<Needs openParen>
        this.closeParen = new Token(")"); // $<Needs openParen>

        this.body = new TableConstructor();
    }
}

// optexport "enum" Name ["(" [params] ")"] "{" {enumfield fieldsep} [".."] "}"
class EnumDecl extends GlobStat {
    constructor() {
        super("EnumDecl");
        this.export = new Export();
        this.enumKw = new Token("enum");
        this.name = new Name();

        this.openParen = new OptToken("(");
        this.params = new DelimitedList("TypedParam"); // $<Needs openParen>
        this.closeParen = new Token(")"); // $<Needs openParen>

        this.openBrace = new Token("{");
        this.fields = new DelimitedList("EnumField");
        this.spread = new OptToken("..");
        this.closeBrace = new Token("}");
    }
}

// optexport [safety] ["struct"] "fn" Name "(" [params] ")" ["->" basicExpr] ["{" block "}"]
class FunctionDecl extends GlobStat {
    constructor() {
        super("FunctionDecl");
        this.export = new Export();
        this.safety = new Safety();
        this.structKw = new OptToken("struct");
        this.fnKw = new Token("fn");
        this.name = new Name();
        this.openParen = new Token("(");
        this.params = new DelimitedList("TypedParam");
        this.closeParen = new Token(")");
        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>
        this.body = new OptBlockNode();
    }
}

// optexport "trait" Name ["(" params ")"] [whereClauses] tableconstructor
class TraitDecl extends GlobStat {
    constructor() {
        super("TraitDecl");
        this.export = new Export();
        this.traitKw = new Token("trait");
        this.name = new Name();

        this.openParen = new OptToken("(");
        this.params = new DelimitedList("TypedParam"); // $<Needs openParen>
        this.closeParen = new Token(")"); // $<Needs openParen>

        this.where = new Expr(); // WhereClauses representation //TODO: no
        this.body = new TableConstructor();
    }
}

// safety externInfo "{" {globstat} "}"
class ExternBlock extends GlobStat {
    constructor() {
        super("ExternBlock");
        this.safety = new Safety();
        this.externKw = new Token("extern");
        this.abiName = new Str();
        this.openBrace = new Token("{");
        this.stats = []; // Array of GlobStat
        this.closeBrace = new Token("}");
    }
}

// optexport ["unsafe"] "impl" ["(" params ")"] [basicExpr "for"] basicExpr [whereClauses] tableconstructor
class ImplDecl extends GlobStat {
    constructor() {
        super("ImplDecl");
        this.export = new Export();
        this.safety = new Safety();
        this.implKw = new Token("impl");

        this.openParen = new OptToken("(");
        this.params = new DelimitedList("TypedParam"); // $<Needs openParen>
        this.closeParen = new Token(")"); // $<Needs openParen>

        this.traitType = new Expr(); // $<Needs forKw>
        this.forKw = new OptToken("for");

        this.targetType = new Expr();

        this.where = new Expr(); // TODO: no
        this.body = new TableConstructor();
    }
}

// optexport "use" modpath useVariant
class UseDecl extends GlobStat {
    constructor() {
        super("UseDecl");
        this.export = new Export();
        this.useKw = new Token("use");
        this.path = new ModPath();
        this.variant = new Expr(); // "as" Name or modpathindex //TODO: no
    }
}

// optexport "mod" Name ["{" chunk "}"]
class ModDecl extends GlobStat {
    constructor() {
        super("ModDecl");
        this.export = new Export();
        this.modKw = new Token("mod");
        this.name = new Name();
        this.openBrace = new OptToken("{");
        this.chunk = []; // Array of globstat // $<Needs openBrace>
        this.closeBrace = new Token("}"); // $<Needs openBrace>
    }
}

// optexport "const" uncondDestrPat "=" expr
class ConstDecl extends GlobStat {
    constructor() {
        super("ConstDecl");
        this.export = new Export();
        this.constKw = new Token("const");
        this.pattern = new UncondDestrPat();
        this.eq = new Token("=");
        this.value = new Expr();
    }
}

// optexport "union" Name ["(" [params] ")"] tableconstructor
class UnionDecl extends GlobStat {
    constructor() {
        super("UnionDecl");
        this.export = new Export();
        this.unionKw = new Token("union");
        this.name = new Name();

        this.openParen = new OptToken("(");
        this.params = new DelimitedList("TypedParam"); // $<Needs openParen>
        this.closeParen = new Token(")"); // $<Needs openParen>

        this.body = new TableConstructor();
    }
}

// ============================================================================
// STATEMENTS (stat)
// ============================================================================

// [label] "loop" ["->" basicExpr] "{" block "}"
class LoopStat extends Stat {
    constructor() {
        super("LoopStat");
        this.labelStart = new OptToken("'");
        this.label = new Name(); // $<Needs labelStart>
        this.labelColon = new Token(":"); // $<Needs labelStart>

        this.loopKw = new Token("loop");
        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>
        this.body = new BlockNode();
    }
}

// [label] "while" basicExpr "{" block "}"
class WhileStat extends Stat {
    constructor() {
        super("WhileStat");
        this.labelStart = new OptToken("'");
        this.label = new Name(); // $<Needs labelStart>
        this.labelColon = new Token(":"); // $<Needs labelStart>

        this.whileKw = new Token("while");
        this.condition = new Expr();
        this.body = new BlockNode();
    }
}

// [label] "for" ["const"] uncondDestrPat "in" basicExpr "{" block "}"
class ForStat extends Stat {
    constructor() {
        super("ForStat");
        this.labelStart = new OptToken("'");
        this.label = new Name(); // $<Needs labelStart>
        this.labelColon = new Token(":"); // $<Needs labelStart>

        this.forKw = new Token("for");
        this.constKw = new OptToken("const");
        this.pattern = new UncondDestrPat();
        this.inKw = new Token("in");
        this.iterable = new Expr();
        this.body = new BlockNode();
    }
}

// "let" uncondDestrPat ["=" expr]
class LetStat extends Stat {
    constructor() {
        super("LetStat");
        this.letKw = new Token("let");
        this.pattern = new UncondDestrPat();
        this.eq = new OptToken("=");
        this.value = new Expr(); // $<Needs eq>
    }
}

// "if" basicExpr blockOrRet {"else" "if" basicExpr blockOrRet} ["else" blockOrRet]
class IfStat extends Stat {
    constructor() {
        super("IfStat");
        this.ifKw = new Token("if");
        this.condition = new Expr();
        this.consequent = new Stat(); // blockOrRet //TODO: blockorret
        this.alternates = []; // { elseKw: Token, ifKw: Token, cond: Expr, body: Stat }
        this.elseKw = new OptToken("else");
        this.elseBlock = new Stat(); // blockOrRet // $<Needs elseKw>
    }
}

// "match" basicExpr matchtypeblock
class MatchStat extends Stat {
    constructor() {
        super("MatchStat");
        this.matchKw = new Token("match");
        this.argument = new Expr();
        this.block = new MatchTypeBlock();
    }
}

// "drop" expr
class DropStat extends Stat {
    constructor() {
        super("DropStat");
        this.dropKw = new Token("drop");
        this.expr = new Expr();
    }
}

// [label] "{" block "}"
class BlockStat extends Stat {
    constructor() {
        super("BlockStat");
        this.labelStart = new OptToken("'");
        this.label = new Name(); // $<Needs labelStart>
        this.labelColon = new Token(":"); // $<Needs labelStart>
        this.block = new BlockNode();
    }
}

// ";"
class EmptyStat extends Stat {
    constructor() {
        super("EmptyStat");
        this.semicol = new Token(";");
    }
}

// ============================================================================
// EXPRESSIONS (expr, basicExpr)
// ============================================================================

// "(" expr ")"
class ParenExpr extends Expr {
    constructor() {
        super("ParenExpr");
        this.openParen = new Token("(");
        this.expr = new Expr();
        this.closeParen = new Token(")");
    }
}

class ModPathExpr extends Expr {
    constructor() {
        super("ModPathExpr");
        this.root = new Name(); //TODO: self, crate
        this.path = []; // Array of {sep: Token("::"), idx: Name}
    }
}

class StrExpr extends Expr {
    constructor() {
        super("StrExpr");
        this.raw = new Str();
    }
}
class NumExpr extends Expr {
    constructor() {
        super("NumExpr");
        this.raw = new Num();
    }
}

class BinExpr extends Expr {
    constructor() {
        super("BinExpr");
        this.left = new Expr();
        this.op = new Token(""); //TODO: no
        this.right = new Expr();
    }
}

class UnaryExpr extends Expr {
    constructor() {
        super("UnaryExpr");
        this.preOps = []; // Array of Token|UnOp
        this.primary = new Expr();
        this.sufOps = []; // Array of Token|UnOp
    }
}

class CallOp extends UnOp {
    constructor() {
        super("CallOp");
        this.dot = new OptToken(".");
        this.method = new Name(); // $<Needs dot>
        this.args = new Args();
    }
}
class DotOp extends UnOp {
    constructor() {
        super("DotOp");
        this.dot = new Token(".");
        this.field = new TuplableName();
    }
}
class ConstDotOp extends UnOp {
    constructor() {
        super("ConstDotOp");
        this.dot = new Token(".:");
        this.field = new Name();
    }
}
class IdxOp extends UnOp {
    constructor() {
        super("IdxOp");
        this.leftBracket = new Token("[");
        this.idx = new Expr();
        this.rightBracket = new Token("]");
    }
}
// "match" basicExpr matchtypeblock
class MatchExpr extends Expr {
    constructor() {
        super("MatchExpr");
        this.matchKw = new Token("match");
        this.argument = new Expr();
        this.block = new MatchTypeBlock();
    }
}
// [label] "loop" ["->" basicExpr] "{" block "}"
class LoopExpr extends Expr {
    constructor() {
        super("LoopExpr");
        this.labelStart = new OptToken("'");
        this.label = new Name(); // $<Needs labelStart>
        this.labelColon = new Token(":"); // $<Needs labelStart>

        this.loopKw = new Token("loop");
        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>
        this.body = new BlockNode();
    }
}

class FnExpr extends Expr {
    constructor() {
        super("FnExpr");
        this.safety = new Safety();
        this.fnKw = new Token("fn");
        this.openParen = new Token("(");
        this.selfParams = new Expr();//TODO: no
        this.params = new DelimitedList("TypedParam");
        this.closeParen = new Token(")");
        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>
        this.body = new OptBlockNode();
    }
}

class LambdaExpr extends Expr {
    constructor() {
        super("LambdaExpr");
        this.safety = new Safety();
        this.pipe1 = new Token("|");
        this.params = new DelimitedList("TypedParam");
        this.pipe2 = new Token("|");

        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>
        this.doubleArrow = new Token("=>"); // $<Needs retArrow>

        this.body = new Expr();
    }
}

// [label] ["const"] "do" ["->" basicExpr] "{" block "}"
class DoExpr extends Expr {
    constructor() {
        super("DoExpr");
        this.labelStart = new OptToken("'");
        this.label = new Name(); // $<Needs labelStart>
        this.labelColon = new Token(":"); // $<Needs labelStart>

        this.constKw = new OptToken("const");
        this.doKw = new Token("do");

        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>

        this.block = new BlockNode();
    }
}
class ConstExpr extends Expr {
    constructor() {
        super("ConstExpr");
        this.constKw = new Token("const");
        this.openParen = new Token("(");
        this.expr = new Expr();
        this.closeParen = new Token(")");
    }
}
