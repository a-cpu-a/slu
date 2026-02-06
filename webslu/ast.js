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

class Export extends CompoundNode {
    constructor() {
        super("Export");
        this.kw = new OptToken("ex");
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

class DelimitedListItem extends CompoundNode {
    constructor() {
        super("DelimitedListItem");
        this.value = null; // The actual node (e.g., Expr, TypedParam)
        this.sep = new OptToken(); // "," or ";"
    }
}

class DelimitedList extends CompoundNode {
    constructor(type) {
        super("DelimitedList");
        this.itemType = type;
        this.items = []; // Array of DelimitedListItem
    }
}

class OptToken extends Node {
    constructor(txt) {
        super("OptToken");
        this.txt = txt;
        this.present = false; // true if parsed
    }
}

class OptExpr extends CompoundNode {
    constructor() {
        super("OptExpr");
        this.expr = new Expr();
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
class OptTableConstructor extends CompoundNode {
    constructor() {
        super("OptTableConstructor");
        this.present = false; // true if it exists
        this.tbl = new TableConstructor();
    }
}

// ============================================================================
// ANNOTATIONS
// ============================================================================

class Annotation extends CompoundNode {
    constructor(type) { super(type); }
}

class ModPathAnnotation extends Annotation {
    constructor() {
        super("ModPathAnnotation");
        this.at = new Token("@");
        this.path = new ModPath();
        this.table = new OptTableConstructor();
    }
}

class DocLineAnnotation extends Annotation {
    constructor() {
        super("DocLineAnnotation");
        this.txt = new Token("---");
        this.content = "";
    }
}

class OuterAnnotation extends CompoundNode {
    constructor(type) { super(type); }
}

class OuterModPathAnnotation extends OuterAnnotation {
    constructor() {
        super("OuterModPathAnnotation");
        this.at = new Token("@<");
        this.path = new ModPath();
        this.table = new OptTableConstructor();
    }
}

class OuterDocLineAnnotation extends OuterAnnotation {
    constructor() {
        super("OuterDocLineAnnotation");
        this.txt = new Token("--<");
        this.content = "";
    }
}

// ============================================================================
// BLOCK OR RETURN (Helper for IfStat)
// ============================================================================

// blockOrRet ::= retstat [";"] | "{" block "}"
class BlockOrRet extends CompoundNode {
    constructor(type) { super(type); }
}

class BlockOrRetBlock extends BlockOrRet {
    constructor() {
        super("BlockOrRetBlock");
        this.block = new BlockNode();
    }
}

class BlockOrRetRetStat extends BlockOrRet {
    constructor() {
        super("BlockOrRetRetStat");
        this.retStat = new RetStat();
        this.semicol = new OptToken(";");
    }
}

// ============================================================================
// PATTERNS (sPat, dPat, uncondDestrPat)
// ============================================================================

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
        this.ops = []; // Array of TypePreOp
    }
}
class Pat extends CompoundNode {
    constructor(type) { super(type); }
}
class SimplePat extends Pat {
    constructor() {
        super("SimplePat");
        this.expr = new Expr();
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

// ============================================================================
// PREOPS
// ============================================================================

class RefAttrs extends CompoundNode {
    constructor() {
        super("RefAttrs");
        this.addrspace = null; // OptToken("in" Name)
        this.lifetime = null;  // OptLifetime
        this.refType = new RefType();
    }
}
class RefType extends CompoundNode {
    constructor(type) { super(type); }
}
class MutRefType extends RefType {
    constructor() { super("MutRefType"); this.kw = new Token("mut"); }
}
class ShareRefType extends RefType {
    constructor() { super("ShareRefType"); this.kw = new Token("share"); }
}
class ConstRefType extends RefType {
    constructor() { super("ConstRefType"); this.kw = new Token("const"); }
}

class PreOp extends CompoundNode {
    constructor(type) { super(type); }
}
class TypePreOp extends PreOp {
    constructor(type) { super(type); }
}

class RefTypePreOp extends TypePreOp {
    constructor() {
        super("RefTypePreOp");
        this.star = new Token("*");
        this.attrs = new RefAttrs();
    }
}

class SlicePreOp extends PreOp {
    constructor() {
        super("SlicePreOp");
        this.open = new Token("[");
        this.close = new Token("]");
    }
}
class DynPreOp extends PreOp {
    constructor() { super("DynPreOp"); this.kw = new Token("dyn"); }
}
class ImplPreOp extends PreOp {
    constructor() { super("ImplPreOp"); this.kw = new Token("impl"); }
}
class UnionPreOp extends PreOp {
    constructor() { super("UnionPreOp"); this.kw = new Token("union"); }
}
class RefPreOp extends PreOp {
    constructor() {
        super("RefPreOp");
        this.amp = new Token("&");
        this.attrs = new RefAttrs();
    }
}
class RangePreOp extends PreOp {
    constructor() { super("RangePreOp"); this.op = new Token(".."); }
}
class AnnotationPreOp extends PreOp {
    constructor() {
        super("AnnotationPreOp");
        this.annotation = new Annotation();
    }
}
class IfPreOp extends PreOp {
    constructor() {
        super("IfPreOp");
        this.ifKw = new Token("if");
        this.expr = new Expr();
        this.arrow = new Token("=>");
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

class BlockNode extends CompoundNode {
    constructor() {
        super("BlockNode");
        this.openBrace = new Token("{");
        this.stats = [];
        this.retStat = new RetStat();
        this.closeBrace = new Token("}");
    }
}

class ModPath extends CompoundNode {
    constructor() {
        super("ModPath");
        this.root = new ModPathRoot();
        this.segments = []; // Array of ModPathSegment
    }
}

class ModPathRoot extends CompoundNode {
    constructor(type) { super(type); }
}
class ModPathRootName extends ModPathRoot {
    constructor() {
        super("ModPathRootName");
        this.name = new Name();
    }
}
class ModPathRootSelf extends ModPathRoot {
    constructor() {
        super("ModPathRootSelf");
        this.kw = new Token("self");
    }
}
class ModPathRootCrate extends ModPathRoot {
    constructor() {
        super("ModPathRootCrate");
        this.kw = new Token("crate");
    }
}
class ModPathRootOp extends ModPathRoot {
    constructor() {
        super("ModPathRootOp");
        this.op = new Token(":>");
    }
}

// "::" Name
class ModPathSegment extends CompoundNode {
    constructor() {
        super("ModPathSegment");
        this.sep = new Token("::");
        this.name = new Name();
    }
}

class MatchItem extends CompoundNode {
    constructor() {
        super("MatchItem");
        this.pat = new Pat();

        this.ifKw = new OptToken("if");
        this.ifExpr = new Expr(); // $<Needs ifKw>

        this.arrow = new Token("=>");
        this.expr = new Expr();
    }
}
class MatchTypeBlock extends CompoundNode {
    constructor() {
        super("MatchTypeBlock");

        this.retArrow = new OptToken("->");
        this.retType = new Expr(); // $<Needs retArrow>

        this.openBrace = new Token("{");
        this.items = new DelimitedList("MatchItem");
        this.closeBrace = new Token("}");
    }
}

class TypedParam extends CompoundNode {
    constructor() {
        super("TypedParam");
        this.constKw = new OptToken("const");
        this.name = new Name();
        this.eq = new Token("=");
        this.type = new Expr();
    }
}

// ============================================================================
// TABLE CONSTRUCTOR
// ============================================================================

class Field extends CompoundNode {
    constructor(type) { super(type); }
}

class NamedField extends Field {
    constructor() {
        super("NamedField");
        this.name = new Name();
        this.eq = new Token("=");
        this.expr = new Expr();
    }
}

class ExprField extends Field {
    constructor() {
        super("UnnamedField");
        this.expr = new Expr();
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

// ============================================================================
// GLOBAL DECLARATIONS (globstat)
// ============================================================================

class GlobStat extends CompoundNode {
    constructor(type) { super(type); }
}

// globstat ::= ";"
class EmptyGlobStat extends GlobStat {
    constructor() {
        super("EmptyGlobStat");
        this.semicol = new Token(";");
    }
}

// optexport "struct" Name ["(" params ")"] tableconstructor
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

// optexport "enum" Name ["(" params ")"] "{" {enumfield fieldsep} [".."] "}"
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

class EnumField extends CompoundNode {
    constructor() {
        super("EnumField");
        this.annotations = []; // Array of Annotation
        this.export = new Export();
        this.name = new Name();
        this.table = new OptTableConstructor(); // TableConstructor
        this.outerAnnotations = []; // Array of OuterAnnotation
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

        this.where = new WhereClauses();
        this.body = new TableConstructor();
    }
}

// safety extern LiteralString "{" {globstat} "}"
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

        this.where = new WhereClauses();
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
        this.variant = new UseVariant();
    }
}

class UseVariant extends CompoundNode {
    constructor(type) { super(type); }
}

class UseAs extends UseVariant {
    constructor() {
        super("UseAs");
        this.asKw = new Token("as");
        this.name = new Name();
    }
}

class SimpleUseVariant extends UseVariant {
    constructor() {
        super("SimpleUseVariant");
    }
}

// "::*"
class StarUseVariant extends UseVariant {
    constructor() {
        super("StarUseVariant");
        this.kw = new Token("::*");
    }
}

// "::{" ("self"|Name) {fieldsep Name} [fieldsep] "}"
class BraceUseVariant extends UseVariant {
    constructor() {
        super("BraceUseVariant");
        this.colonColon = new Token("::");
        this.openBrace = new Token("{");

        this.selfKw = new OptToken("self");
        this.selfDelim = new Token(""); // $<Needs selfKw>

        this.items = new DelimitedList("Name");
        this.closeBrace = new Token("}");
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

// optexport "union" Name ["(" params ")"] tableconstructor
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

class RetStat extends CompoundNode {
    constructor(type) { super(type); }
}

class ReturnStat extends RetStat {
    constructor() {
        super("ReturnStat");
        this.returnKw = new Token("return");
        this.expr = new OptExpr();
    }
}

class BreakStat extends RetStat {
    constructor() {
        super("BreakStat");
        this.breakKw = new Token("break");
        this.label = new OptToken("'");
        this.labelName = new Name(); // $<Needs label>
        this.expr = new OptExpr();
    }
}

class ContinueStat extends RetStat {
    constructor() {
        super("ContinueStat");
        this.continueKw = new Token("continue");
        this.label = new OptToken("'");
        this.labelName = new Name(); // $<Needs label>
        this.expr = new OptExpr();
    }
}

class ThrowStat extends RetStat {
    constructor() {
        super("ThrowStat");
        this.throwKw = new Token("throw");
        this.expr = new Expr();
    }
}

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

// Represents the "else if ... " part of an IfStat
class IfAlternate extends CompoundNode {
    constructor() {
        super("IfAlternate");
        this.elseKw = new Token("else");
        this.ifKw = new Token("if");
        this.condition = new Expr();
        this.body = new BlockOrRet();
    }
}
// "if" basicExpr blockOrRet {"else" "if" basicExpr blockOrRet} ["else" blockOrRet]
class IfStat extends Stat {
    constructor() {
        super("IfStat");
        this.ifKw = new Token("if");
        this.condition = new Expr();
        this.consequent = new BlockOrRet();
        this.alternates = []; // Array of IfAlternate
        this.elseKw = new OptToken("else");
        this.elseBlock = new BlockOrRet();
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

// "unsafe" "{" {stat} "}"
class UnsafeStat extends Stat {
    constructor() {
        super("UnsafeStat");
        this.unsafeKw = new Token("unsafe");
        this.openBrace = new Token("{");
        this.stats = []; // Array of Stat
        this.closeBrace = new Token("}");
    }
}

// var "=" expr
class AssignStat extends Stat {
    constructor() {
        super("AssignStat");
        this.var = new Var();
        this.eq = new Token("=");
        this.expr = new Expr();
    }
}

// var selfablecall
class CallStat extends Stat {
    constructor() {
        super("CallStat");
        this.var = new Var();
        this.call = new SelfableCall();
    }
}

// "TODO!" LiteralString
class TodoStat extends Stat {
    constructor() {
        super("TodoStat");
        this.kw = new Token("TODO!");
        this.msg = new Str();
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
        this.path = new ModPath();
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
        this.op = new Token("");
        this.right = new Expr();
    }
}

class UnaryExpr extends Expr {
    constructor() {
        super("UnaryExpr");
        this.PreOps = []; // Array of PreOp
        this.primary = new Expr();
        this.sufOps = []; // Array of SufOp
    }
}

class SufOp extends CompoundNode {
    constructor(type) { super(type); }
}

// var ::= (Name | "(" expr ")") {{selfablecall} subvar}
class Var extends CompoundNode {
    constructor() {
        super("Var");
        this.root = null; // Name or ParenExpr
        this.suffixes = []; // Array of SubVar | SelfableCall
    }
}

class SubVar extends SufOp {
    constructor() { super("SubVar"); }
}
class DerefSubVar extends SubVar {
    constructor() { super("DerefSubVar"); this.op = new Token(".*"); }
}
class DotSubVar extends SubVar {
    constructor() { super("DotSubVar"); this.op = new Token("."); this.field = new TuplableName(); }
}
class ConstDotSubVar extends SubVar {
    constructor() { super("ConstDotSubVar"); this.op = new Token(".:"); this.field = new Name(); }
}
class IdxSubVar extends SubVar {
    constructor() { super("IdxSubVar"); this.open = new Token("["); this.expr = new Expr(); this.close = new Token("]"); }
}

class SelfableCall extends SufOp {
    constructor() {
        super("SelfableCall");
        this.dot = new OptToken(".");
        this.method = new Name(); // $<Needs dot>
        this.args = new Args();
    }
}

class TryOp extends SufOp {
    constructor() {
        super("TryOp");
        this.tryKw = new Token("try");
        this.block = new MatchTypeBlock();
    }
}

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

        this.selfParamRefs = []; // Array of RefTypePreOp // $<Needs selfParam>
        this.selfParam = new OptToken("self");
        this.selfParamDelim = new Token(""); // $<Needs selfParam>

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

class TodoExpr extends Expr {
    constructor() {
        super("TodoExpr");
        this.kw = new Token("TODO!");
        this.msg = new Str();
    }
}

class UnderscoreExpr extends Expr {
    constructor() {
        super("UnderscoreExpr");
        this.us = new Token("_");
    }
}

class LifetimeExpr extends Expr {
    constructor() {
        super("LifetimeExpr");
        this.names = []; // Array of {kw:Token("/"),l:Name}
    }
}

class UnboundedRangeExpr extends Expr {
    constructor() {
        super("UnboundedRangeExpr");
        this.op = new Token("..");
    }
}

// "_COMP_TODO!" "(" LiteralString "," expr ")"
class CompTodoExpr extends Expr {
    constructor() {
        super("CompTodoExpr");
        this.kw = new Token("_COMP_TODO!");
        this.openParen = new Token("(");
        this.msg = new Str(); // LiteralString
        this.comma = new Token(",");
        this.expr = new Expr();
        this.closeParen = new Token(")");
    }
}

// ============================================================================
// UTILS
// ============================================================================

class WhereClauses extends CompoundNode {
    constructor() {
        super("WhereClauses");
        this.whereKw = new Token("where");
        this.clauses = new DelimitedList("WhereClause");
    }
}

class WhereClause extends CompoundNode {
    constructor() {
        super("WhereClause");
        this.name = new Name(); // or "Self"
        this.colon = new Token(":");
        this.type = new Expr();
    }
}







/**
 * Visitor implementation for the provided AST.
 * 
 * Features:
 * 1. Generic Traversal: Automatically visits all child nodes.
 * 2. Mutation: Supports replacing nodes in-place or returning new nodes.
 * 3. Scope Management: Built-in stack for symbol tables (useful for name resolution).
 * 
 * Extend this class to create specific passes (e.g., TypeChecker, ConstEvaluator).
 */
class Visitor {
    constructor() {
        // Stack of scopes for symbol resolution
        this.scopeStack = [];
    }

    // ============================================================
    // MAIN ENTRY POINT
    // ============================================================

    /**
     * Visits an AST node.
     * If a specific method 'visit<NodeType>' exists, it is called.
     * Otherwise, defaults to generic traversal of children.
     * 
     * @param {Node|CompoundNode|null} node 
     * @returns {Node|CompoundNode|null} The original node, a mutated node, or a replacement node.
     */
    visit(node) {
        if (!node) return null;

        // Only process objects (Nodes)
        if (typeof node !== 'object') return node;

        // Dispatch to specific handler if defined
        const methodName = 'visit' + node.type;
        if (this[methodName] && typeof this[methodName] === 'function') {
            const result = this[methodName](node);
            // If the visitor returns a value, assume it is the new node (mutation/replacement).
            // If it returns undefined, assume it mutated in-place and return the original node.
            return result !== undefined ? result : node;
        }

        // Default behavior: recurse into children
        return this.visitChildren(node);
    }

    // ============================================================
    // GENERIC TRAVERSAL ENGINE
    // ============================================================

    /**
     * Recursively visits all properties of a node.
     * Handles arrays (of nodes) and single node references.
     * It attempts to update the parent references if a child node is replaced.
     */
    visitChildren(node) {
        if (!node || typeof node !== 'object') return node;

        for (const key in node) {
            if (!node.hasOwnProperty(key)) continue;

            // Skip metadata and helper properties
            if (key === 'type' || key === 'preSpace' || key === 'itemType') continue;

            const value = node[key];

            // 1. Handle Arrays (e.g., stats, params, items)
            if (Array.isArray(value)) {
                for (let i = 0; i < value.length; i++) {
                    const item = value[i];
                    // Check if item is an AST Node
                    if (item && (item instanceof Node || item instanceof CompoundNode)) {
                        const newItem = this.visit(item);
                        // Mutation Support: Replace array item if the visitor returned a new object
                        if (newItem !== item) {
                            value[i] = newItem;
                        }
                    }
                }
            }
            // 2. Handle Single Nodes (e.g., expr, body, condition)
            else if (value && (value instanceof Node || value instanceof CompoundNode)) {
                // Optimization: If the node is an "Opt" wrapper (OptExpr, OptToken, etc.)
                // and it is not present, skip visiting its inner dummy node.
                if (value.hasOwnProperty('present') && value.present === false) {
                    continue;
                }

                const newValue = this.visit(value);
                // Mutation Support: Replace property if the visitor returned a new object
                if (newValue !== value) {
                    node[key] = newValue;
                }
            }
        }
        return node;
    }

    // ============================================================
    // SCOPE & SYMBOL MANAGEMENT HELPERS
    // ============================================================

    /** Returns the current active scope */
    get currentScope() {
        return this.scopeStack[this.scopeStack.length - 1];
    }

    /** Enters a new scope (e.g., entering a Block or Function) */
    enterScope(node) {
        const scope = {
            node: node,
            parent: this.currentScope,
            symbols: {}, // name -> { type, node, kind }
            types: {}    // name -> TypeNode
        };
        this.scopeStack.push(scope);
    }

    /** Exits the current scope */
    leaveScope() {
        if (this.scopeStack.length > 0) {
            this.scopeStack.pop();
        }
    }

    /** Defines a symbol in the current scope */
    defineSymbol(name, metadata) {
        if (this.currentScope) {
            this.currentScope.symbols[name] = metadata;
        } else {
            console.warn(`Defining symbol ${name} outside of a scope`);
        }
    }

    /** Looks up a symbol in the scope stack */
    resolveSymbol(name) {
        for (let i = this.scopeStack.length - 1; i >= 0; i--) {
            const scope = this.scopeStack[i];
            if (scope.symbols[name]) {
                return scope.symbols[name];
            }
        }
        return null;
    }

    // ============================================================
    // BASE STRUCTURAL VISITORS (OVERRIDE FOR SCOPE LOGIC)
    // ============================================================

    // Override these to inject scope management automatically

    visitBlockNode(node) {
        this.enterScope(node);
        this.visitChildren(node);
        this.leaveScope();
        return node;
    }

    visitFunctionDecl(node) {
        this.enterScope(node);
        this.visitChildren(node);
        this.leaveScope();
        return node;
    }

    visitImplDecl(node) {
        this.enterScope(node);
        this.visitChildren(node);
        this.leaveScope();
        return node;
    }

    visitForStat(node) {
        this.enterScope(node);
        this.visitChildren(node);
        this.leaveScope();
        return node;
    }

    // ============================================================
    // EXAMPLE: CONST EVALUATOR / INLINING
    // ============================================================

    /*
    // Example of how to extend the visitor for Const Inlining
    
    visitModPathExpr(node) {
        const pathStr = this.modPathToString(node.path);
        const sym = this.resolveSymbol(pathStr);
        
        // If symbol is a constant and has a known evaluated value
        if (sym && sym.kind === 'const' && sym.value) {
            // Return the value node directly (Inlining)
            // We clone it to avoid mutating the original definition
            return this.cloneNode(sym.value);
        }
        
        return node;
    }
    
    visitBinExpr(node) {
        this.visitChildren(node);
        
        // Simple constant folding example: 1 + 2 -> 3
        if (node.left.type === 'NumExpr' && node.right.type === 'NumExpr') {
            if (node.op.txt === '+') {
                const l = parseFloat(node.left.raw.num);
                const r = parseFloat(node.right.raw.num);
                const newNode = new NumExpr(new Num((l + r).toString()));
                newNode.preSpace = node.preSpace;
                return newNode;
            }
        }
        return node;
    }
    */
}



// ============================================================================
// PARSER IMPLEMENTATION
// ============================================================================

class Parser {
    constructor(input) {
        this.input = input;
        this.pos = 0;
        this.len = input.length;
        this.tokens = [];
        this.tokPos = 0;
        this.tokenize();
    }

    // ========================================================================
    // LEXER / TOKENIZER
    // ========================================================================

    tokenize() {
        // Define Keywords based on spec
        const keywords = new Set([
            "drop", "trait", "any", "has", "raw", "glob", "reloc", "nosize", "concept", "nostride",
            "unstrided", "at", "of", "asm", "box", "out", "auto", "case", "only", "pure", "test",
            "with", "wrap", "final", "inout", "mixin", "become", "impure", "Inject", "inject",
            "inline", "typeof", "Modify", "Shadow", "Capture", "default", "discard", "uniform",
            "unsized", "virtual", "abstract", "override", "Operation", "operation", "Overwrite",
            "overwrite", "groupshare", "groupshared", "threadlocal", "gen", "copy", "move", "async",
            "await", "super", "yield", "static", "generator", "as", "ex", "fn", "it", "dyn", "let",
            "mod", "mut", "rec", "try", "use", "also", "enum", "impl", "loop", "safe", "self",
            "Self", "TODO", "alloc", "axiom", "catch", "const", "crate", "defer", "macro", "match",
            "share", "throw", "trans", "union", "where", "extern", "struct", "unsafe", "continue",
            "_COMP_TODO", "do", "if", "in", "or", "and", "end", "for", "else", "goto", "then",
            "break", "local", "until", "while", "elseif", "global", "repeat", "return", "function"
        ]);

        // Define Symbols based on spec
        // Using a Set for O(1) lookup. We will check for length 3, then 2, then 1.
        const symbols = new Set([
            "|||", ">=<", "===", "<=>", "+++", "//=", "///", "***", "**-", "...", ":::", "---",
            "~~", "~|", "~>", "~=", "~<", "~+", "~^", "~%", "~&", "~/", "~@", "~?", "~!", "~-", "~",
            "||", "|>", "|=", "|<", "|+", "|^", "|%", "|&", "|/", "|@", "|?", "|!", "|-", "|",
            ">~", ">|", ">>", ">=", "><", ">+", ">^", ">%", ">&", ">/", ">*", ">@", ">?", ">!", ">",
            "=<", "=",
            "<~", "<|", "<>", "<=", "<<", "<+", "<^", "<%", "<&", "</", "<*", "<@", "<?", "<!", "<",
            "+~", "+|", "+>", "+=", "+<", "++", "+^", "+%", "+/", "+*", "+@", "+!", "+-", "+",
            "^~", "^|", "^>", "^=", "^<", "^+", "^^", "^%", "^/", "^*", "^@", "^?", "^!", "^",
            "%~", "%|", "%>", "%=", "%<", "%+", "%^", "%%", "%/", "%*", "%@", "%?", "%!", "%",
            "&~", "&>", "&=", "&<", "&+", "&^", "&%", "&&", "&*", "&?", "&",
            "/~", "/|", "/>", "/=", "/<", "/+", "/^", "/%", "/&", "//", "/*", "/@", "/?", "/!", "/",
            "*~", "*|", "*>", "*=", "*<", "*+", "*^", "*%", "*/", "**", "*?", "*!", "*",
            "@<", "@@", "@", "}", "{", "]", "[", ")", "(", "\"", "'", ".*", "..", ".:", ".", "?&", "?@", "?!", "?",
            "!~", "!|", "!>", "!=", "!<", "!+", "!^", "!%", "!/", "!*", "!@", "!?", "!!", "!",
            ":>", ":",
            ";", ",",
            "-~", "-|", "->", "-=", "-<", "-+", "-^", "-%", "-/", "-*", "-@", "-?", "--", "-",
            "__", "_"
        ]);

        // Helper to handle Lua-style long brackets
        const handleLongBracket = (isComment) => {
            const start = this.pos;
            this.pos++; // Skip '['

            const openBracketPos = this.pos;
            let level = 0;

            // Count '=' signs
            while (this.pos < this.len && this.input[this.pos] === '=') {
                this.pos++;
                level++;
            }

            // Expect closing '['
            if (this.pos >= this.len || this.input[this.pos] !== '[') {
                this.pos = start; // Reset and let standard logic handle it
                //TODO: soft error if in comments
                return false;
            }
            this.pos++; // Consume the second '['

            const openSeq = this.input.substring(openBracketPos, this.pos);
            const closeSeq = openSeq.replace(/\[/g, ']'); // [==[ becomes ]==]

            // Scan for closing sequence
            while (this.pos + closeSeq.length <= this.len) {
                if (this.input.substring(this.pos, this.pos + closeSeq.length) === closeSeq) {
                    this.pos += closeSeq.length;
                    return true; // Success
                }
                this.pos++;
            }
            this.pos = this.len;
            //TODO: soft error
            return true;
        };

        let savedStart = null;
        while (this.pos < this.len) {
            const start = (savedStart != null) ? savedStart : this.pos;
            savedStart = null;

            let ch = this.input[this.pos];
            let preSpace = "";

            // Skip whitespace and capture it
            while (/\s/.test(ch) && this.pos < this.len) {
                this.pos++;
                ch = this.input[this.pos];
            }
            if (start !== this.pos) {
                preSpace = this.input.substring(start, this.pos);
            }

            if (this.pos >= this.len) break; // TODO: final preSpace must be stored in the ast, prob a new field in the file or something

            const tokenStart = this.pos;

            // Comments: --...
            if (ch === '-' && this.pos + 1 < this.len && this.input[this.pos + 1] === '-') {
                this.pos += 2;
                const third = (this.pos < this.len) ? this.input[this.pos] : null;
                // Doc Comments / LineOfText: --- or --<
                if (third === '-' || third === '<') {
                    this.pos++;
                    this.tokens.push({
                        type: 'Symbol',
                        txt: "--" + third,
                        preSpace: preSpace
                    });
                    if (this.pos < this.len) {
                        ch = this.input[this.pos];
                        // Long String Literal: [[...]] or [=[...]=]
                        if (ch === '[') {
                            // We check if next char is [ or =
                            if (this.pos + 1 < this.len && (this.input[this.pos + 1] === '[' || this.input[this.pos + 1] === '=')) {
                                if (handleLongBracket(true)) {
                                    this.tokens.push({
                                        type: 'LiteralString',
                                        txt: this.input.substring(tokenStart, this.pos),
                                        preSpace: preSpace
                                    });
                                    continue;
                                }
                            }
                            // If not a long bracket (or malformed), fall through to symbol handling
                        }
                    }

                    let content = "";
                    while (this.pos < this.len && !/[\r\n]/.test(this.input[this.pos])) {
                        content += this.input[this.pos];
                        this.pos++;
                    }
                    this.tokens.push({
                        type: 'LineOfText',
                        txt: content,
                        preSpace: ""
                    });
                    continue;
                }
                // Multiline Comment: --[=[...]=]
                if (third === '[') {
                    if (this.pos + 1 < this.len && (this.input[this.pos + 1] === '[' || this.input[this.pos + 1] === '=')) {
                        if (handleLongBracket(true)) {
                            // Comment is whitespace, so we skip it (do not push token)

                            // Preserve preSpace for the next token
                            savedStart = start;
                            continue;
                        }
                    }
                    // Treat as line comment start if bracket invalid.
                }

                // Regular Single Line Comment: --
                while (this.pos < this.len && !/[\r\n]/.test(this.input[this.pos])) {
                    this.pos++;
                }
                // Preserve preSpace for the next token
                savedStart = start;

                // Whitespace is ignored, so we just continue
                continue;
            }

            // Strings
            if (ch === '"' || ch === "'") {
                const quote = ch;
                this.pos++;
                while (this.pos < this.len && this.input[this.pos] !== quote) {
                    if (this.input[this.pos] === '\\') this.pos++; // skip escape
                    this.pos++;
                }
                if (this.pos < this.len) this.pos++; // consume end quote
                //todo: this also does a error recovery from eof ^^^
                this.tokens.push({ type: 'LiteralString', txt: this.input.substring(tokenStart, this.pos), preSpace: preSpace });
                continue;
            }

            // Long String Literal: [[...]] or [=[...]=]
            if (ch === '[') {
                // We check if next char is [ or =
                if (this.pos + 1 < this.len && (this.input[this.pos + 1] === '[' || this.input[this.pos + 1] === '=')) {
                    if (handleLongBracket(false)) {
                        this.tokens.push({
                            type: 'LiteralString',
                            txt: this.input.substring(tokenStart, this.pos),
                            preSpace: preSpace
                        });
                        continue;
                    }
                }
                // If not a long bracket (or malformed), fall through to symbol handling
            }
            {
                let tmpFound = false;
                // Symbols
                for (let i = 3; i > 0; i--) {
                    if ((this.pos + i - 1) < this.len) {
                        const tok = this.input.substring(this.pos, this.pos + i);
                        if (symbols.has(tok)) {
                            this.pos += i;
                            this.tokens.push({ type: 'Symbol', txt: tok, preSpace: preSpace });
                            tmpFound = true;
                            break;
                        }
                    }
                }
                if (tmpFound)
                    continue;
            }

            // Numbers (Hex or Dec)
            if (/[0-9]/.test(ch)) {
                let tokenStart = this.pos;
                let isHex = false;

                // Helper to parse digit lists based on grammar: digit [{digit|"_"}digit]
                // Returns true if at least one digit was consumed.
                const parseList = (hexMode) => {
                    const digitRegex = hexMode ? /[0-9a-fA-F]/ : /[0-9]/;
                    const start = this.pos;
                    let hasDigit = false;

                    while (this.pos < this.len) {
                        let nch = this.input[this.pos];

                        // Match digit
                        if (digitRegex.test(nch)) {
                            this.pos++;
                            hasDigit = true;
                        }
                        // Match underscore separator
                        else if (nch === '_' && this.pos + 1 < this.len && digitRegex.test(this.input[this.pos + 1])) {
                            //TODO: allow multiple underscores in between digits
                            this.pos++; // Consume underscore, next digit consumed in next iteration
                        }
                        else {
                            break;
                        }
                    }
                    return hasDigit;
                };

                // 1. Check for Hex Prefix (0x or 0X)
                // Grammar: hexStart(hexDigList)
                if (ch === '0' && this.pos + 1 < this.len && /[xX]/.test(this.input[this.pos + 1])) {
                    this.pos += 2; // Consume '0x'
                    // Attempt to parse hex digits. 
                    if (parseList(true)) {
                        isHex = true;
                    } else {
                        let errCh = this.input[this.pos];
                        throw new Error(`Expected hex digits, but found: ${errCh} at ${this.pos}`);
                    }
                }

                // 2. Parse Integer Part
                // If not hex, parse the initial digits.
                if (!isHex) {
                    parseList(false);
                }

                // 3. Parse Fractional Part (Optional)
                // Grammar: ["." digList] or ["." hexDigList]
                if (this.pos < this.len && this.input[this.pos] === '.') {
                    let nextChar = this.input[this.pos + 1];
                    let digitRegex = isHex ? /[0-9a-fA-F]/ : /[0-9]/;

                    if (nextChar && digitRegex.test(nextChar)) {
                        this.pos++; // Consume '.'
                        parseList(isHex);
                    }
                }

                // 4. Parse Exponent Part (Optional)
                // Grammar: [("e"|"E")[expSign]digList] or [("p"|"P")[expSign]hexDigList]
                if (this.pos < this.len) {
                    let current = this.input[this.pos];
                    let isExp = isHex ? /[pP]/.test(current) : /[eE]/.test(current);

                    if (isExp) {
                        this.pos++; // Consume 'e', 'E', 'p', or 'P'

                        // Optional Sign
                        if (this.pos < this.len && /[+\-]/.test(this.input[this.pos])) {
                            this.pos++;
                        }

                        // Mandatory digits after exponent marker
                        if (parseList(isHex)) {
                            // Success
                        } else {
                            let errCh = this.input[this.pos];
                            throw new Error(`Expected exponent digits, but found: ${errCh} at ${this.pos}`);
                        }
                    }

                    if (this.pos < this.len) {
                        let nextCh = this.input[this.pos];
                        if (/[0-9_a-zA-Z!]/.test(nextCh)) {
                            throw new Error(`Unexpected character: ${nextCh} at ${this.pos}`);
                        }
                    }
                }

                this.tokens.push({ type: 'Numeral', txt: this.input.substring(tokenStart, this.pos), preSpace: preSpace });
                continue;
            }

            // Identifiers / Keywords / Names
            if (/[a-zA-Z_]/.test(ch)) {
                while (this.pos < this.len && /[a-zA-Z0-9_]/.test(this.input[this.pos])) {
                    this.pos++;
                }
                // Macro calls:
                if (this.pos < this.len && this.input[this.pos] == "!") {
                    this.pos++;
                }
                const txt = this.input.substring(tokenStart, this.pos);
                this.tokens.push({
                    type: (keywords.has(txt) || /^_*$/.test(txt)) ? 'Keyword' : 'Name', // `/^_*$/` => made of only underscores
                    txt: txt,
                    preSpace: preSpace
                });
                continue;
            }

            throw new Error(`Unexpected character: ${ch} at ${this.pos}`);
        }

        // EOF
        this.tokens.push({ type: 'EOF', txt: '', preSpace: '' });
    }

    // ========================================================================
    // PARSER UTILS
    // ========================================================================

    peek(offset = 0) {
        return this.tokens[this.tokPos + offset];
    }

    consume(expectedType = null, expectedTxt = null) {
        const tok = this.tokens[this.tokPos];
        if (expectedType && tok.type !== expectedType) return null;
        if (expectedTxt && tok.txt !== expectedTxt) return null;

        this.tokPos++;
        return tok;
    }

    match(...args) {
        return this.consume(...args) !== null;
    }

    expect(type, txt) {
        const tok = this.consume(type, txt);
        if (!tok) {
            const current = this.peek();
            throw new Error(`Expected ${type} "${txt}", found ${current.type} "${current.txt}"`);
        }
        return tok;
    }

    // Helper to set properties on a Node object
    applyToken(node, token, propName) {
        if (token) {
            node[propName] = new Token();
            node[propName].txt = token.txt;
            node[propName].preSpace = token.preSpace;
        }
        return node;
    }

    // ========================================================================
    // PARSING IMPLEMENTATION
    // ========================================================================

    parseChunk() {
        const stats = [];
        while (this.peek().type !== 'EOF') {
            stats.push(this.parseGlobStat());
        }
        return stats;
    }

    // -------------------------------------------------------------------------
    // GlobStats
    // -------------------------------------------------------------------------

    parseGlobStat() {
        const annotations = this.parseAnnotations(false); // false = inner

        // Macro invocation
        if (this.checkMacroInvoke()) {
            return this.parseMacroInvoke();
        }

        // OptExport
        const exportNode = new Export();
        if (this.match('Keyword', 'ex')) {
            exportNode.kw.txt = 'ex';
            exportNode.kw.present = true;
        }

        const peekType = this.peek().type;
        const peekTxt = this.peek().txt;

        if (this.match('Symbol', ';')) {
            const node = new EmptyGlobStat();
            node.semicol.txt = ';';
            return node;
        }

        if (peekTxt === 'enum') {
            return this.parseEnumDecl(exportNode, annotations);
        }

        if (peekTxt === 'struct') {
            return this.parseStructDecl(exportNode, annotations);
        }

        if (peekTxt === 'union') {
            return this.parseUnionDecl(exportNode, annotations);
        }

        if (peekTxt === 'trait') {
            return this.parseTraitDecl(exportNode, annotations);
        }

        if (peekTxt === 'impl') {
            return this.parseImplDecl(exportNode, annotations);
        }

        if (peekTxt === 'use') {
            return this.parseUseDecl(exportNode, annotations);
        }

        if (peekTxt === 'mod') {
            return this.parseModDecl(exportNode, annotations);
        }

        if (peekTxt === 'const') {
            return this.parseConstDecl(exportNode, annotations);
        }

        if (peekTxt === 'axiom') {
            // optexport "axiom" Name ["as" "{" {axiomStat} "}"]
            const node = new CompoundNode("AxiomDecl"); // Placeholder class if not defined
            node.kw = new Token(); node.kw.txt = "axiom";
            this.expect('Keyword', 'axiom');
            node.name = this.parseName();
            if (this.match('Keyword', 'as')) {
                this.expect('Symbol', '{');
                // parse axiom stats
                node.block = [];
                while (!this.match('Symbol', '}')) {
                    node.block.push(this.parseAxiomStat());
                }
            }
            return node;
        }

        if (peekTxt === 'safe' || peekTxt === 'unsafe') {
            // safety extern ...
            const safety = this.parseSafety();
            if (this.peek().txt === 'extern') {
                return this.parseExternBlock(safety, exportNode, annotations);
            }
            // Could be impl or fn starting with safety
        }

        if (peekTxt === 'fn') {
            return this.parseFunctionDecl(exportNode, annotations);
        }

        // If we have annotations but nothing matched, it might be a decorated regular globstat or error
        // For now, assume logic above covers top level keywords

        throw new Error(`Unknown global statement starting with ${peekTxt}`);
    }

    parseStructDecl(exportNode, annotations) {
        const node = new StructDecl();
        node.export = exportNode;
        node.annotations = annotations;
        this.applyToken(node, this.expect('Keyword', 'struct'), 'structKw');
        node.name = this.parseName();

        // Params
        if (this.match('Symbol', '(')) {
            node.openParen.present = true;
            node.openParen.txt = '(';
            node.params = this.parseDelimitedList(() => this.parseTypedParam());
            this.expect('Symbol', ')');
            node.closeParen.txt = ')';
        }

        node.body = this.parseTableConstructor();
        return node;
    }

    parseEnumDecl(exportNode, annotations) {
        const node = new EnumDecl();
        node.export = exportNode;
        node.annotations = annotations;
        this.applyToken(node, this.expect('Keyword', 'enum'), 'enumKw');
        node.name = this.parseName();

        if (this.match('Symbol', '(')) {
            node.openParen.present = true;
            node.openParen.txt = '(';
            node.params = this.parseDelimitedList(() => this.parseTypedParam());
            this.expect('Symbol', ')');
            node.closeParen.txt = ')';
        }

        this.expect('Symbol', '{');
        node.openBrace.txt = '{';
        node.fields = this.parseDelimitedList(() => this.parseEnumField(), true); // true = allow trailing spread?
        if (this.match('Symbol', '..')) {
            node.spread.present = true;
            node.spread.txt = '..';
        }
        this.expect('Symbol', '}');
        node.closeBrace.txt = '}';
        return node;
    }

    parseEnumField() {
        const node = new EnumField();
        node.annotations = this.parseAnnotations(false);
        node.export = new Export(); // Fields can have 'ex'?
        if (this.match('Keyword', 'ex')) {
            node.export.kw.present = true;
        }
        node.name = this.parseName();
        node.table = new OptTableConstructor();
        // Check for table constructor immediately following name
        if (this.peek().txt === '{') {
            node.table.present = true;
            node.table.tbl = this.parseTableConstructor();
        }
        node.outerAnnotations = this.parseAnnotations(true); // Outer
        return node;
    }

    parseFunctionDecl(exportNode, annotations) {
        const node = new FunctionDecl();
        node.export = exportNode;
        node.annotations = annotations;

        // [safety] ["struct"]
        node.safety = this.parseSafety();
        if (this.match('Keyword', 'struct')) {
            node.structKw.present = true;
            node.structKw.txt = 'struct';
        }

        this.applyToken(node, this.expect('Keyword', 'fn'), 'fnKw');
        node.name = this.parseName();

        this.expect('Symbol', '(');
        node.openParen.txt = '(';

        // Self params
        if (this.peek().txt === 'self' || (this.peek().txt === '*' || this.peek().txt === '&')) {
            // Parse selfParams
            while (this.peek().txt === '*' || this.peek().txt === '&') {
                // refAttrs prefix
                // Simplified: assuming *mut self etc.
                // Parsing full refAttrs
                const preOp = this.parseTypePreOp(); // e.g. *
                if (preOp) node.selfParamRefs.push(preOp);
            }
            if (this.match('Keyword', 'self')) {
                node.selfParam.present = true;
                node.selfParam.txt = 'self';
                if (this.match('Symbol', ',') || this.match('Symbol', ';')) {
                    node.selfParamDelim.txt = this.tokens[this.tokPos - 1].txt;
                }
            }
        }

        node.params = this.parseDelimitedList(() => this.parseTypedParam());
        this.expect('Symbol', ')');
        node.closeParen.txt = ')';

        if (this.match('Symbol', '->')) {
            node.retArrow.present = true;
            node.retArrow.txt = '->';
            node.retType = this.parseExpr();
        }

        if (this.peek().txt === '{') {
            node.body.present = true;
            node.body.block = this.parseBlock();
        }

        return node;
    }

    parseTraitDecl(exportNode, annotations) {
        const node = new TraitDecl();
        node.export = exportNode;
        node.annotations = annotations;
        this.applyToken(node, this.expect('Keyword', 'trait'), 'traitKw');
        node.name = this.parseName();

        if (this.match('Symbol', '(')) {
            node.openParen.present = true;
            node.params = this.parseDelimitedList(() => this.parseTypedParam());
            this.expect('Symbol', ')');
            node.closeParen.txt = ')';
        }

        node.where = this.parseWhereClauses();
        node.body = this.parseTableConstructor();
        return node;
    }

    parseImplDecl(exportNode, annotations) {
        const node = new ImplDecl();
        node.export = exportNode;
        node.annotations = annotations;

        // ["unsafe"] "impl"
        // Note: parseSafety consumes 'safe' or 'unsafe'. 
        // But impl grammar: optexport ["unsafe"] "impl". 
        // If we consumed safety before, we must check if next is 'impl'.
        // If not, it might be ExternBlock.
        if (node.safety.kind === 'default' && this.peek().txt === 'unsafe') {
            node.safety.kind = 'unsafe';
            this.consume();
        }

        this.applyToken(node, this.expect('Keyword', 'impl'), 'implKw');

        if (this.match('Symbol', '(')) {
            node.openParen.present = true;
            node.params = this.parseDelimitedList(() => this.parseTypedParam());
            this.expect('Symbol', ')');
            node.closeParen.txt = ')';
        }

        // [basicExpr "for"] basicExpr
        // This is tricky. We need to parse basicExpr. If followed by "for", it's the trait.
        // Otherwise, it's the type.
        // But the grammar says `basicExpr "for" basicExpr`.
        // We can parse the first expr. Then peek for "for".
        const save = this.tokPos;
        try {
            const potentialTrait = this.parseExpr();
            if (this.match('Keyword', 'for')) {
                node.traitType = potentialTrait;
                node.forKw.present = true;
                node.forKw.txt = 'for';
                node.targetType = this.parseExpr();
            } else {
                // Backtrack: the first expr was actually the targetType
                this.tokPos = save;
                node.targetType = this.parseExpr();
            }
        } catch (e) {
            // If parsing failed, maybe it was just the targetType
            this.tokPos = save;
            node.targetType = this.parseExpr();
        }

        node.where = this.parseWhereClauses();
        node.body = this.parseTableConstructor();
        return node;
    }

    parseUseDecl(exportNode, annotations) {
        const node = new UseDecl();
        node.export = exportNode;
        node.annotations = annotations;
        this.applyToken(node, this.expect('Keyword', 'use'), 'useKw');
        node.path = this.parseModPath();
        node.variant = this.parseUseVariant();
        return node;
    }

    parseUseVariant() {
        if (this.match('Keyword', 'as')) {
            const node = new UseAs();
            node.name = this.parseName();
            return node;
        }
        if (this.peek().txt === '::' && this.peek(1).txt === '*') {
            const node = new StarUseVariant();
            this.consume(); // ::
            this.consume(); // *
            return node;
        }
        if (this.peek().txt === '::' && this.peek(1).txt === '{') {
            const node = new BraceUseVariant();
            this.consume(); // ::
            this.expect('Symbol', '{');

            if (this.match('Keyword', 'self')) {
                node.selfKw.present = true;
                if (this.match('Symbol', ',') || this.match('Symbol', ';')) {
                    node.selfDelim.txt = this.tokens[this.tokPos - 1].txt;
                }
            }

            node.items = this.parseDelimitedList(() => this.parseName());
            this.expect('Symbol', '}');
            return node;
        }
        return new SimpleUseVariant();
    }

    parseModDecl(exportNode, annotations) {
        const node = new ModDecl();
        node.export = exportNode;
        node.annotations = annotations;
        this.applyToken(node, this.expect('Keyword', 'mod'), 'modKw');
        node.name = this.parseName();

        if (this.match('Symbol', '{')) {
            node.openBrace.present = true;
            node.chunk = [];
            while (!this.match('Symbol', '}')) {
                node.chunk.push(this.parseGlobStat());
            }
            node.closeBrace.txt = '}';
        }
        return node;
    }

    parseConstDecl(exportNode, annotations) {
        const node = new ConstDecl();
        node.export = exportNode;
        node.annotations = annotations;
        this.applyToken(node, this.expect('Keyword', 'const'), 'constKw');
        node.pattern = this.parseUncondDestrPat();
        this.expect('Symbol', '=');
        node.eq.txt = '=';
        node.value = this.parseExpr();
        return node;
    }

    parseUnionDecl(exportNode, annotations) {
        const node = new UnionDecl();
        node.export = exportNode;
        node.annotations = annotations;
        this.applyToken(node, this.expect('Keyword', 'union'), 'unionKw');
        node.name = this.parseName();

        if (this.match('Symbol', '(')) {
            node.openParen.present = true;
            node.params = this.parseDelimitedList(() => this.parseTypedParam());
            this.expect('Symbol', ')');
            node.closeParen.txt = ')';
        }

        node.body = this.parseTableConstructor();
        return node;
    }

    parseExternBlock(safety, exportNode, annotations) {
        const node = new ExternBlock();
        node.safety = safety;
        node.export = exportNode;
        node.annotations = annotations;
        this.applyToken(node, this.expect('Keyword', 'extern'), 'externKw');
        node.abiName = new Str(); // Parse literal string
        const strTok = this.expect('LiteralString');
        node.abiName.str = strTok.txt;

        this.expect('Symbol', '{');
        node.openBrace.txt = '{';
        node.stats = [];
        while (!this.match('Symbol', '}')) {
            node.stats.push(this.parseGlobStat());
        }
        node.closeBrace.txt = '}';
        return node;
    }

    // -------------------------------------------------------------------------
    // Statements (stat)
    // -------------------------------------------------------------------------

    parseStat() {
        // Check annotations first
        const annotations = this.parseAnnotations(false);

        if (this.match('Symbol', ';')) {
            const node = new EmptyStat();
            node.semicol.txt = ';';
            return node;
        }

        const peekTxt = this.peek().txt;

        // Label
        let labelStart = null;
        let labelName = null;
        if (this.match('Symbol', "'")) {
            // Check if next is Name and then :
            // Label ::= "'" Name ":"
            // Grammar: [label] "loop" ...
            // We need lookahead. If ' followed by Name followed by :
            const save = this.tokPos;
            if (this.peek().type === 'Name' && this.peek(1).txt === ':') {
                labelStart = "'";
                labelName = this.parseName().name;
                this.expect('Symbol', ':');
            } else {
                this.tokPos = save;
            }
        }

        if (peekTxt === 'loop') {
            return this.parseLoopStat(labelStart, labelName, annotations);
        }

        if (peekTxt === 'while') {
            return this.parseWhileStat(labelStart, labelName, annotations);
        }

        if (peekTxt === 'for') {
            return this.parseForStat(labelStart, labelName, annotations);
        }

        if (peekTxt === 'if') {
            return this.parseIfStat(annotations);
        }

        if (peekTxt === 'unsafe') {
            const node = new UnsafeStat();
            node.annotations = annotations;
            this.applyToken(node, this.expect('Keyword', 'unsafe'), 'unsafeKw');
            this.expect('Symbol', '{');
            node.openBrace.txt = '{';
            node.stats = [];
            while (!this.match('Symbol', '}')) {
                node.stats.push(this.parseStat());
            }
            node.closeBrace.txt = '}';
            return node;
        }

        if (peekTxt === 'let') {
            const node = new LetStat();
            node.annotations = annotations;
            this.applyToken(node, this.expect('Keyword', 'let'), 'letKw');
            node.pattern = this.parseUncondDestrPat();
            if (this.match('Symbol', '=')) {
                node.eq.present = true;
                node.eq.txt = '=';
                node.value = this.parseExpr();
            }
            return node;
        }

        if (peekTxt === 'drop') {
            const node = new DropStat();
            node.annotations = annotations;
            this.applyToken(node, this.expect('Keyword', 'drop'), 'dropKw');
            node.expr = this.parseExpr();
            return node;
        }

        if (peekTxt === 'return') {
            const node = new ReturnStat();
            node.annotations = annotations;
            this.applyToken(node, this.expect('Keyword', 'return'), 'returnKw');
            if (!this.checkStatEnd()) {
                node.expr = this.parseExpr();
                node.expr.present = true;
            }
            return node;
        }

        if (peekTxt === 'break') {
            const node = new BreakStat();
            node.annotations = annotations;
            this.applyToken(node, this.expect('Keyword', 'break'), 'breakKw');
            if (this.match('Symbol', "'")) {
                node.label.present = true;
                node.label.txt = "'";
                node.labelName = this.parseName();
            }
            if (!this.checkStatEnd()) {
                node.expr = this.parseExpr();
                node.expr.present = true;
            }
            return node;
        }

        if (peekTxt === 'continue') {
            const node = new ContinueStat();
            node.annotations = annotations;
            this.applyToken(node, this.expect('Keyword', 'continue'), 'continueKw');
            if (this.match('Symbol', "'")) {
                node.label.present = true;
                node.label.txt = "'";
                node.labelName = this.parseName();
            }
            if (!this.checkStatEnd()) {
                node.expr = this.parseExpr();
                node.expr.present = true;
            }
            return node;
        }

        if (peekTxt === 'throw') {
            const node = new ThrowStat();
            node.annotations = annotations;
            this.applyToken(node, this.expect('Keyword', 'throw'), 'throwKw');
            node.expr = this.parseExpr();
            return node;
        }

        if (peekTxt === 'match') {
            return this.parseMatchStat(annotations);
        }

        if (peekTxt === '{') {
            const node = new BlockStat();
            node.annotations = annotations;
            if (labelStart) {
                node.labelStart.present = true;
                node.labelStart.txt = "'";
                node.label.name = labelName;
                node.labelColon.txt = ':';
            }
            node.block = this.parseBlock();
            return node;
        }

        // Macro invocation
        if (this.checkMacroInvoke()) {
            const mi = this.parseMacroInvoke();
            // MacroInvocation can be a statement.
            // Assuming we wrap it in a generic MacroStat node if needed, or just return it.
            // Since GlobStat can be macroInvSpl, and Stat includes GlobStat, returning it is fine.
            return mi;
        }

        // var "=" expr | var selfablecall
        // Need to disambiguate.
        // var = ...
        // var (call)
        // If it is a Name, we look ahead.
        // If we see =, it's assign. If we see ( or . or .: or [, it might be call or suffix.
        // But "var selfablecall" is a statement.
        // So: var ( ... ) or var . ... ( ... )

        const save = this.tokPos;
        try {
            const v = this.parseVar();
            if (this.match('Symbol', '=')) {
                const node = new AssignStat();
                node.annotations = annotations;
                node.var = v;
                node.eq.txt = '=';
                node.expr = this.parseExpr();
                return node;
            } else {
                // Must be call or suffix. 
                // The grammar says: var selfablecall.
                // selfablecall ::= [dotOp Name] (basicArgs | tableconstructor)
                // But parseVar already consumes suffixes including calls.
                // This is contradictory or overlapping.
                // "var" rule: (Name | "(" expr ")") {{selfablecall} subvar}
                // This means var includes calls.
                // So "var selfablecall" means a var followed by ANOTHER call?
                // e.g. `foo.bar()()` ? No, that's just var.
                // Maybe `var` is strictly the LHS part, and `selfablecall` is the function call part?
                // No, `var` includes `selfablecall`.
                // Let's assume `var` parses the whole chain.
                // If we are here, we didn't match `=`.
                // Is it a statement?
                // If the var chain ends with a call, it's a CallStat.
                // Check if the last element in var.suffixes is a SelfableCall.
                // If not, maybe it's invalid as a statement unless it's an expression statement?
                // `expr` is not in `stat`. `stat` -> `statOrExpr`.
                // `statOrExpr` -> `loop`, `match`, `TODO!`. (Not generic expr).
                // So generic expressions are NOT statements.
                // So it MUST be AssignStat or CallStat.
                // If no `=`, we look for `selfablecall` at the end.

                // Actually, parseVar is greedy. If I parse `foo()`, it's a var with a call suffix.
                // The grammar `var selfablecall` suggests `var` is the target, `selfablecall` is the invocation.
                // Perhaps `var` does NOT consume the final `selfablecall`?
                // `var ::= (Name | "(" expr ")") {{selfablecall} subvar}`
                // This includes `selfablecall` in the loop.
                // Okay, let's look at `selfablecall` rule: `[dotOp Name] (basicArgs | tableconstructor)`.
                // If I have `foo()`, `var` parses it.
                // If I have `foo = 1`, `var` parses `foo`.
                // So `var selfablecall` statement likely handles cases where we treat the parsed var as a call.

                // Let's simplify: if parsed var has a Call suffix at the end, it's a CallStat.
                const node = new CallStat();
                node.annotations = annotations;
                node.var = v;
                // We might need to extract the call part if var ate it all, 
                // but the class structure separates `var` and `call`.
                // I will assume `parseVar` parses the LHS of an assignment or the object of a call.
                // If it's a call stat, we expect `var` to be the object, then we parse `selfablecall`.
                // So `parseVar` should stop before the call? 
                // `var` rule allows `selfablecall`.
                // Let's adjust `parseVar` to be `parseLValue`.
                // But strict adherence to grammar:
                // `stat ::= var selfablecall`
                // This implies `parseVar` parses `Name` (or paren), then `subvar`s that are NOT `selfablecall`?
                // `selfablecall` is a `sufop`. `subvar` is a `sufop`.
                // They are mixed in the loop `{{selfablecall} subvar}`.
                // This means `var` can be `foo.bar.baz()`.
                // So `var` IS the call expression.
                // Then `var selfablecall` in stat means `foo.bar.baz` followed by `()`.
                // So `parseVar` should stop *before* a call if we are in a CallStat context?
                // This context sensitivity is hard.
                // Easier path: Parse full expression. If it looks like a call, wrap in CallStat.

                // Actually, let's look at `expr`. `expr` -> `sufop` -> `selfablecall`.
                // So expressions handle calls.
                // Statement `var selfablecall` is redundant if `expr` was a statement, but it's not.
                // So I will implement `parseVar` to parse the full chain.
                // Then check if the last thing was a call.

                // Re-reading: `stat ::= var selfablecall`.
                // This suggests `var` does NOT include the final call.
                // `var ::= ... {{selfablecall} subvar}`.
                // Okay, I will parse `var` such that it parses everything *except* if the grammar for stat demands splitting.
                // Given the AST class `CallStat` has `var` and `call`, I must split them.
                // So `parseVar` should parse up to the point where a call starts?
                // Or parse everything and then split?
                // Let's try parsing `var` as the `Name` (or paren) followed by `subvar` (access/index) but NOT `basicArgs`?
                // `selfablecall` uses `basicArgs` or `tableconstructor`.
                // `subvar` uses `.*` `.` `[` `]`.
                // So `var` = Root + (Access/Index)*.
                // `stat` `var selfablecall` = (Root + Access*) + Call.
                // This makes sense!

                // Retrying with this logic:
                this.tokPos = save; // Reset
                const lval = this.parseVarLValue(); // Name + subvar (no calls)

                // Now check for selfablecall
                const call = this.parseSelfableCall(); // Can be null?
                if (call) {
                    const cs = new CallStat();
                    cs.annotations = annotations;
                    cs.var = lval;
                    cs.call = call;
                    return cs;
                }

                // If no call, maybe it was an assignment we missed? 
                // No, we checked for '=' earlier.
                // Maybe it's just `var`? Not allowed in stat grammar unless globstat or macro.
                throw new Error("Expected call or assignment");
            }
        } catch (e) {
            this.tokPos = save;
        }

        // Fallback to statOrExpr? No, stat includes statOrExpr.
        return this.parseStatOrExpr(annotations);
    }

    // Helper to parse Name/Paren + subvars (no calls)
    parseVarLValue() {
        const node = new Var();
        if (this.peek().txt === '(') {
            this.consume();
            node.root = this.parseExpr();
            this.expect('Symbol', ')');
        } else {
            node.root = this.parseName();
        }

        // subvar ::= ".*" | dotOp tupleableName | "[" expr "]"
        // Note: `selfablecall` is excluded here based on previous logic deduction.
        while (true) {
            if (this.match('Symbol', '.*')) {
                const op = new DerefSubVar();
                op.op.txt = '.*';
                node.suffixes.push(op);
            } else if (this.peek().txt === '.' || this.peek().txt === '.:') {
                const op = this.peek().txt === '.' ? new DotSubVar() : new ConstDotSubVar();
                op.op.txt = this.consume().txt;
                // tupleableName ::= Name | integral
                if (/[0-9]/.test(this.peek().txt) || (this.peek().txt === '0' && /[xX]/.test(this.peek(1).txt))) {
                    op.field = new NumTuplableName();
                    op.field.name = this.parseNum();
                } else {
                    op.field = new NameTuplableName();
                    op.field.name = this.parseName();
                }
                node.suffixes.push(op);
            } else if (this.match('Symbol', '[')) {
                const op = new IdxSubVar();
                op.open.txt = '[';
                op.expr = this.parseExpr();
                this.expect('Symbol', ']');
                op.close.txt = ']';
                node.suffixes.push(op);
            } else {
                break;
            }
        }
        return node;
    }

    parseSelfableCall() {
        // [dotOp Name] (basicArgs | tableconstructor)
        const node = new SelfableCall();

        // Optional dot name
        if (this.peek().txt === '.' || this.peek().txt === '.:') {
            node.dot.present = true;
            node.dot.txt = this.consume().txt;
            node.method = this.parseName();
        }

        // Args
        if (this.peek().txt === '(') {
            node.args = new ParenArgs();
            this.expect('Symbol', '(');
            node.args.args = this.parseDelimitedList(() => this.parseExpr());
            this.expect('Symbol', ')');
        } else if (this.peek().type === 'LiteralString') {
            node.args = new StrArgs();
            const tok = this.consume();
            node.args.args.str = tok.txt;
        } else if (this.peek().type === 'Numeral') {
            node.args = new NumArgs();
            const tok = this.consume();
            node.args.args.num = tok.txt; // string in Num class? AST says Num has `num` property
            // Convert to Num object
            node.args.args = this.parseNumFromTok(tok);
        } else if (this.peek().txt === '{') {
            node.args = new TableArgs();
            node.args.args = this.parseTableConstructor();
        } else {
            return null;
        }
        return node;
    }

    parseLoopStat(labelStart, labelName, annotations) {
        const node = new LoopStat();
        node.annotations = annotations;
        if (labelStart) {
            node.labelStart.present = true;
            node.labelStart.txt = "'";
            node.label.name = labelName;
            node.labelColon.txt = ':';
        }
        this.applyToken(node, this.expect('Keyword', 'loop'), 'loopKw');
        if (this.match('Symbol', '->')) {
            node.retArrow.present = true;
            node.retArrow.txt = '->';
            node.retType = this.parseExpr();
        }
        node.body = this.parseBlock();
        return node;
    }

    parseWhileStat(labelStart, labelName, annotations) {
        const node = new WhileStat();
        node.annotations = annotations;
        if (labelStart) {
            node.labelStart.present = true;
            node.labelStart.txt = "'";
            node.label.name = labelName;
            node.labelColon.txt = ':';
        }
        this.applyToken(node, this.expect('Keyword', 'while'), 'whileKw');
        node.condition = this.parseExpr();
        node.body = this.parseBlock();
        return node;
    }

    parseForStat(labelStart, labelName, annotations) {
        const node = new ForStat();
        node.annotations = annotations;
        if (labelStart) {
            node.labelStart.present = true;
            node.labelStart.txt = "'";
            node.label.name = labelName;
            node.labelColon.txt = ':';
        }
        this.applyToken(node, this.expect('Keyword', 'for'), 'forKw');

        if (this.match('Keyword', 'const')) {
            node.constKw.present = true;
            node.constKw.txt = 'const';
        }

        node.pattern = this.parseUncondDestrPat();
        this.expect('Keyword', 'in');
        node.inKw.txt = 'in';
        node.iterable = this.parseExpr();
        node.body = this.parseBlock();
        return node;
    }

    parseIfStat(annotations) {
        const node = new IfStat();
        node.annotations = annotations;
        this.applyToken(node, this.expect('Keyword', 'if'), 'ifKw');
        node.condition = this.parseExpr();
        node.consequent = this.parseBlockOrRet();

        // Else if chain
        while (this.match('Keyword', 'else')) {
            if (this.match('Keyword', 'if')) {
                const alt = new IfAlternate();
                alt.elseKw.txt = 'else';
                alt.ifKw.txt = 'if';
                alt.condition = this.parseExpr();
                alt.body = this.parseBlockOrRet();
                node.alternates.push(alt);
            } else {
                node.elseKw.present = true;
                node.elseKw.txt = 'else';
                node.elseBlock = this.parseBlockOrRet();
                break;
            }
        }
        return node;
    }

    parseMatchStat(annotations) {
        const node = new MatchStat();
        node.annotations = annotations;
        this.applyToken(node, this.expect('Keyword', 'match'), 'matchKw');
        node.argument = this.parseExpr();
        node.block = this.parseMatchTypeBlock();
        return node;
    }

    parseStatOrExpr(annotations) {
        // [label] "loop" ...
        // [label] "match" ...
        // "TODO!" ...

        let labelStart = null;
        let labelName = null;
        if (this.match('Symbol', "'")) {
            const save = this.tokPos;
            if (this.peek().type === 'Name' && this.peek(1).txt === ':') {
                labelName = this.parseName().name;
                this.expect('Symbol', ':');
                labelStart = "'";
            } else {
                this.tokPos = save;
            }
        }

        if (this.peek().txt === 'loop') {
            return this.parseLoopStat(labelStart, labelName, annotations);
        }

        if (this.peek().txt === 'match') {
            return this.parseMatchStat(annotations); // Match doesn't take label in grammar? statOrExpr says [label] loop... match...
            // Wait, `statOrExpr ::= [label] "loop" ... | "match" ...`
            // So match has no label in grammar line provided, but stat says `[label] "match"`?
            // Grammar check:
            // stat ::= ... | statOrExpr | ...
            // statOrExpr ::= [label] "loop" ... | "match" ...
            // Stat has explicit `[label] "match"`? No.
            // Stat has: `[label] "while" ...`
            // It seems match has no label in this grammar version.
            return this.parseMatchStat(annotations);
        }

        if (this.peek().txt === 'TODO!') {
            const node = new TodoStat();
            node.annotations = annotations;
            node.kw.txt = 'TODO!';
            this.consume();
            node.msg = new Str();
            node.msg.str = this.expect('LiteralString').txt;
            return node;
        }

        // If nothing matches, it might be a block? `[label] "{" block "}"` is blockstat
        if (this.peek().txt === '{') {
            return this.parseBlockStat(annotations, labelStart, labelName);
        }

        throw new Error("Expected statOrExpr");
    }

    parseBlockStat(annotations, labelStart, labelName) {
        const node = new BlockStat();
        node.annotations = annotations;
        if (labelStart) {
            node.labelStart.present = true;
            node.labelStart.txt = "'";
            node.label.name = labelName;
            node.labelColon.txt = ':';
        }
        node.block = this.parseBlock();
        return node;
    }

    // -------------------------------------------------------------------------
    // Expressions (expr, basicExpr)
    // -------------------------------------------------------------------------

    parseExpr() {
        return this.parseExprPrecedence(0);
    }

    parseExprPrecedence(prec) {
        let left = this.parseUnaryExpr();

        while (true) {
            const opTok = this.peekBinOp();
            if (!opTok) break;

            const nextPrec = this.getPrecedence(opTok.txt);
            if (nextPrec < prec) break;

            this.consume();
            const op = opTok.txt;
            const nextP = this.isRightAssoc(op) ? nextPrec : nextPrec + 1;
            const right = this.parseExprPrecedence(nextP);

            const node = new BinExpr();
            node.left = left;
            node.op.txt = op;
            node.right = right;
            left = node;
        }
        return left;
    }

    peekBinOp() {
        const t = this.peek();
        if (t.type !== 'Symbol' && t.type !== 'Keyword') return null;
        const valid = new Set(['+', '-', '*', '/', '//', '^', '%', '++', '<', '<=', '>', '>=', '==', '!=', 'and', 'or', '~', '|', '..', 'else', '**', 'as']);
        if (valid.has(t.txt)) return t;
        return null;
    }

    getPrecedence(op) {
        // Custom precedence table
        if (op === 'as') return 100;
        if (op === '**') return 90;
        if (op === '*' || op === '/' || op === '//' || op === '%') return 80;
        if (op === '+' || op === '-') return 70;
        if (op === '..') return 60;
        if (op === '<' || op === '<=' || op === '>' || op === '>=' || op === '==' || op === '!=') return 50;
        if (op === 'and') return 30;
        if (op === 'or') return 20;
        if (op === 'else') return 10;
        return 0;
    }

    isRightAssoc(op) {
        return op === '**' || op === 'else'; // assume else (ternary) is right assoc
    }

    parseUnaryExpr() {
        const ops = [];
        while (this.peekPreOp()) {
            ops.push(this.parsePreOp());
        }

        let atom = this.parseBasicExpr();

        // Suffixes
        const suffixes = [];
        while (true) {
            const suf = this.parseSufOp();
            if (!suf) break;
            suffixes.push(suf);
        }

        if (ops.length > 0 || suffixes.length > 0) {
            const node = new UnaryExpr();
            node.PreOps = ops;
            node.primary = atom;
            node.sufOps = suffixes;
            return node;
        }
        return atom;
    }

    parseBasicExpr() {
        const t = this.peek();

        if (t.txt === '(') {
            const node = new ParenExpr();
            this.consume();
            node.expr = this.parseExpr();
            this.expect('Symbol', ')');
            return node;
        }

        if (t.type === 'LiteralString') {
            const node = new StrExpr();
            node.raw.str = this.consume().txt;
            return node;
        }

        if (t.type === 'Numeral') {
            const node = new NumExpr();
            node.raw = this.parseNumFromTok(this.consume());
            return node;
        }

        if (t.txt === '{') {
            return this.parseTableConstructor();
        }

        if (t.txt === '_') {
            const node = new UnderscoreExpr();
            this.consume();
            return node;
        }

        if (t.txt === '/' && this.peek(1).type === 'Name') {
            // Lifetime
            const node = new LifetimeExpr();
            while (this.match('Symbol', '/')) {
                const n = this.parseName();
                node.names.push({ kw: new Token(), l: n }); // Populate token
                node.names[node.names.length - 1].kw.txt = '/';
            }
            return node;
        }

        if (t.txt === '..' && this.peek(1).txt !== '.') {
            const node = new UnboundedRangeExpr();
            this.consume();
            return node;
        }

        if (t.txt === 'TODO!') {
            const node = new TodoExpr();
            this.consume();
            node.msg.str = this.expect('LiteralString').txt;
            return node;
        }

        if (t.txt === '_COMP_TODO!') {
            const node = new CompTodoExpr();
            this.consume();
            this.expect('Symbol', '(');
            node.msg.str = this.expect('LiteralString').txt;
            this.expect('Symbol', ',');
            node.expr = this.parseExpr();
            this.expect('Symbol', ')');
            return node;
        }

        if (t.txt === 'const' && this.peek(1).txt === '(') {
            const node = new ConstExpr();
            this.consume();
            this.expect('Symbol', '(');
            node.expr = this.parseExpr();
            this.expect('Symbol', ')');
            return node;
        }

        // Lambda or Closure
        if (t.txt === 'safe' || t.txt === 'unsafe' || t.txt === '|') {
            // safety "|" ...
            const node = new LambdaExpr();
            if (t.txt !== '|') {
                node.safety.kind = this.consume().txt;
            }
            this.expect('Symbol', '|');
            node.params = this.parseDelimitedList(() => this.parseTypedParam());
            this.expect('Symbol', '|');

            if (this.match('Symbol', '->')) {
                node.retArrow.present = true;
                node.retArrow.txt = '->';
                node.retType = this.parseExpr();
            }
            this.expect('Symbol', '=>');
            node.doubleArrow.txt = '=>';
            node.body = this.parseExpr();
            return node;
        }

        // Fn Expr
        if (t.txt === 'safe' || t.txt === 'unsafe' || t.txt === 'fn') {
            const node = new FnExpr();
            if (t.txt !== 'fn') {
                node.safety.kind = this.consume().txt;
            }
            this.applyToken(node, this.expect('Keyword', 'fn'), 'fnKw');
            this.expect('Symbol', '(');

            // selfParams
            if (this.peek().txt === '*' || this.peek().txt === '&') {
                node.selfParamRefs.push(this.parseTypePreOp());
            }
            if (this.match('Keyword', 'self')) {
                node.selfParam.present = true;
                if (this.match('Symbol', ',') || this.match('Symbol', ';')) {
                    node.selfParamDelim.txt = this.tokens[this.tokPos - 1].txt;
                }
            }

            node.params = this.parseDelimitedList(() => this.parseTypedParam());
            this.expect('Symbol', ')');

            if (this.match('Symbol', '->')) {
                node.retArrow.present = true;
                node.retType = this.parseExpr();
            }

            if (this.peek().txt === '{') {
                node.body.present = true;
                node.body.block = this.parseBlock();
            }
            return node;
        }

        // Do Expr
        if (t.txt === 'do' || (t.type === 'Name' && this.peek(1).txt === 'do')) { // Label
            // [label] ["const"] "do" ...
            const node = new DoExpr();

            if (this.match('Symbol', "'")) {
                node.labelStart.present = true;
                node.labelStart.txt = "'";
                node.label.name = this.parseName().name;
                this.expect('Symbol', ':');
            }

            if (this.match('Keyword', 'const')) {
                node.constKw.present = true;
                node.constKw.txt = 'const';
            }

            this.applyToken(node, this.expect('Keyword', 'do'), 'doKw');
            if (this.match('Symbol', '->')) {
                node.retArrow.present = true;
                node.retArrow.txt = '->';
                node.retType = this.parseExpr();
            }
            node.block = this.parseBlock();
            return node;
        }

        // statOrExpr (loop, match, TODO!)
        // Only if not handled above (match/loop/TODO can be expressions)
        // The logic for `statOrExpr` is partially in `parseStat`.
        // But `loop` and `match` are in `statOrExpr`.
        // Since `match` and `loop` start with keywords, we can check here.

        // Note: `parseStat` handles `statOrExpr`. 
        // But we are in `basicExpr` -> `statOrExpr`.
        // So we can call `parseStatOrExpr(null)` (no extra annotations).
        // However, `parseStat` returns `Stat` which is not an `Expr`.
        // `LoopExpr` and `MatchExpr` exist.
        // `LoopStat` exists.
        // We need to instantiate Expr classes.

        if (t.txt === 'loop') {
            return this.parseLoopExpr();
        }
        if (t.txt === 'match') {
            return this.parseMatchExpr();
        }

        // ModPath (Name, self, crate, :>)
        if (t.type === 'Name' || t.txt === 'self' || t.txt === 'crate' || t.txt === ':>') {
            return this.parseModPathExpr();
        }

        throw new Error(`Unexpected token in expression: ${t.txt}`);
    }

    parseLoopExpr() {
        // Copy logic from LoopStat but return LoopExpr
        const node = new LoopExpr();
        if (this.match('Symbol', "'")) {
            node.labelStart.present = true;
            node.labelStart.txt = "'";
            node.label.name = this.parseName().name;
            this.expect('Symbol', ':');
        }
        this.applyToken(node, this.expect('Keyword', 'loop'), 'loopKw');
        if (this.match('Symbol', '->')) {
            node.retArrow.present = true;
            node.retType = this.parseExpr();
        }
        node.body = this.parseBlock();
        return node;
    }

    parseMatchExpr() {
        const node = new MatchExpr();
        this.applyToken(node, this.expect('Keyword', 'match'), 'matchKw');
        node.argument = this.parseExpr();
        node.block = this.parseMatchTypeBlock();
        return node;
    }

    // -------------------------------------------------------------------------
    // Components & Helpers
    // -------------------------------------------------------------------------

    parsePreOp() {
        const t = this.peek();
        if (t.txt === '-' || t.txt === '!' || t.txt === 'ex' || t.txt === 'mut' || t.txt === 'dyn' || t.txt === 'impl' || t.txt === 'union' || t.txt === '~' || t.txt == '..') {
            const node = new PreOp();
            node.type = 'SimplePreOp';
            node.op = new Token(); // Hack: generic preop
            node.op.txt = this.consume().txt;
            return node;
        }

        if (t.txt === '*') {
            return this.parseTypePreOp();
        }
        if (t.txt === '&') {
            const node = new RefPreOp();
            this.consume();
            node.attrs = this.parseRefAttrs();
            return node;
        }
        if (t.txt === '[' && this.peek(1).txt === ']') {
            const node = new SlicePreOp();
            this.consume(); // [
            this.consume(); // ]
            return node;
        }
        if (t.txt === '@' || t.txt === '---') {
            const node = new AnnotationPreOp();
            node.annotation = this.parseAnnotation(); // Might be inner or outer? Assuming inner
            return node;
        }
        if (t.txt === 'if') {
            const node = new IfPreOp();
            this.consume();
            node.expr = this.parseExpr();
            this.expect('Symbol', '=>');
            return node;
        }

        return null;
    }

    parseTypePreOp() {
        if (this.peek().txt === '*') {
            const node = new RefTypePreOp();
            this.consume();
            node.attrs = this.parseRefAttrs();
            return node;
        }
        return null;
    }

    parseRefAttrs() {
        const node = new RefAttrs();
        // addrspace
        if (this.match('Keyword', 'in')) {
            node.addrspace = this.parseName();
        }
        // lifetime
        if (this.peek().txt === '/') {
            // Parse list of /Name
            node.lifetime = [];
            while (this.match('Symbol', '/')) {
                node.lifetime.push(this.parseName());
            }
        }
        // refType
        if (this.match('Keyword', 'const')) {
            node.refType = new ConstRefType();
        } else if (this.match('Keyword', 'share')) {
            node.refType = new ShareRefType();
        } else if (this.match('Keyword', 'mut')) {
            node.refType = new MutRefType();
        }
        return node;
    }

    parseSufOp() {
        // subvar | [dotOp Name] basicArgs | .. | ? | tryOp

        if (this.peek().txt === '.*') {
            const node = new DerefSubVar();
            this.consume();
            return node;
        }

        if (this.peek().txt === '.' || this.peek().txt === '.:') {
            const isDot = this.peek().txt === '.';
            const node = isDot ? new DotSubVar() : new ConstDotSubVar();
            this.consume();
            node.field = new NameTuplableName(); // Only Name allowed in dot suffix of expr?
            // Grammar says `dotOp Name` for suffix. `dotOp` is `.` or `.:`
            node.field.name = this.parseName();
            return node;
        }

        if (this.peek().txt === '[') {
            const node = new IdxSubVar();
            this.consume();
            node.expr = this.parseExpr();
            this.expect('Symbol', ']');
            return node;
        }

        if (this.peek().txt === '..' && this.peek(1).txt !== '.') {
            const node = new RangePreOp(); // Or RangeSufOp? AST uses RangePreOp
            this.consume();
            return node;
        }

        if (this.peek().txt === '?') {
            const node = new SufOp();
            node.type = 'QuestionOp'; // Placeholder
            this.consume();
            return node;
        }

        if (this.peek().txt === 'try') {
            const node = new TryOp();
            this.consume();
            node.block = this.parseMatchTypeBlock();
            return node;
        }

        // Call
        if (this.peek().txt === '(' || this.peek().type === 'LiteralString' || this.peek().type === 'Numeral' || this.peek().txt === '{') {
            // Check if it has a dot name before it?
            // `[dotOp Name] basicArgs`.
            // If dot name exists, it's part of the suffix.
            const node = new SelfableCall();
            if (this.peek().txt === '.' || this.peek().txt === '.:') {
                node.dot.present = true;
                node.dot.txt = this.consume().txt;
                node.method = this.parseName();
            }

            // Args
            if (this.peek().txt === '(') {
                node.args = new ParenArgs();
                this.expect('Symbol', '(');
                node.args.args = this.parseDelimitedList(() => this.parseExpr());
                this.expect('Symbol', ')');
            } else if (this.peek().type === 'LiteralString') {
                node.args = new StrArgs();
                const tok = this.consume();
                node.args.args.str = tok.txt;
            } else if (this.peek().type === 'Numeral') {
                node.args = new NumArgs();
                node.args.args = this.parseNumFromTok(this.consume());
            } else if (this.peek().txt === '{') {
                node.args = new TableArgs();
                node.args.args = this.parseTableConstructor();
            }
            return node;
        }

        return null;
    }

    parseBlock() {
        const node = new BlockNode();
        this.expect('Symbol', '{');
        while (!this.match('Symbol', '}')) {
            // Check for return stat at end?
            // block ::= {stat} [retstat [";"]]
            // We look ahead. If it is return/break/continue/throw, it's a retstat.
            const t = this.peek().txt;
            if (t === 'return' || t === 'break' || t === 'continue' || t === 'throw') {
                node.retStat = this.parseRetStat();
                if (this.match('Symbol', ';')) {
                    // consume optional semicolon
                }
                // After retstat, block must end (or error). Grammar says [retstat] is last.
                if (this.peek().txt !== '}') {
                    throw new Error("Expected end of block after return statement");
                }
            } else {
                node.stats.push(this.parseStat());
            }
        }
        return node;
    }

    parseBlockOrRet() {
        if (this.peek().txt === '{') {
            const node = new BlockOrRetBlock();
            node.block = this.parseBlock();
            return node;
        }
        const node = new BlockOrRetRetStat();
        node.retStat = this.parseRetStat();
        if (this.match('Symbol', ';')) {
            node.semicol.present = true;
            node.semicol.txt = ';';
        }
        return node;
    }

    parseRetStat() {
        const t = this.peek().txt;
        if (t === 'return') return this.parseReturnStat();
        if (t === 'break') return this.parseBreakStat();
        if (t === 'continue') return this.parseContinueStat();
        if (t === 'throw') return this.parseThrowStat();
        throw new Error("Expected return statement");
    }

    parseReturnStat() {
        const node = new ReturnStat();
        this.consume();
        if (!this.checkStatEnd()) {
            node.expr = this.parseExpr();
            node.expr.present = true;
        }
        return node;
    }

    parseBreakStat() {
        const node = new BreakStat();
        this.consume();
        if (this.match('Symbol', "'")) {
            node.label.present = true;
            node.labelName = this.parseName();
        }
        if (!this.checkStatEnd()) {
            node.expr = this.parseExpr();
            node.expr.present = true;
        }
        return node;
    }

    parseContinueStat() {
        const node = new ContinueStat();
        this.consume();
        if (this.match('Symbol', "'")) {
            node.label.present = true;
            node.labelName = this.parseName();
        }
        if (!this.checkStatEnd()) {
            node.expr = this.parseExpr();
            node.expr.present = true;
        }
        return node;
    }

    parseThrowStat() {
        const node = new ThrowStat();
        this.consume();
        node.expr = this.parseExpr();
        return node;
    }

    checkStatEnd() {
        return this.peek().txt === ';' || this.peek().txt === '}' || this.peek().txt === 'else' || this.peek().txt === 'EOF';
    }

    parseMatchTypeBlock() {
        const node = new MatchTypeBlock();
        if (this.match('Symbol', '->')) {
            node.retArrow.present = true;
            node.retArrow.txt = '->';
            node.retType = this.parseExpr();
        }
        this.expect('Symbol', '{');
        node.items = this.parseDelimitedList(() => this.parseMatchItem());
        this.expect('Symbol', '}');
        return node;
    }

    parseMatchItem() {
        const node = new MatchItem();
        node.pat = this.parsePat();
        if (this.match('Keyword', 'if')) {
            node.ifKw.present = true;
            node.ifKw.txt = 'if';
            node.ifExpr = this.parseExpr();
        }
        this.expect('Symbol', '=>');
        node.expr = this.parseExpr();
        return node;
    }

    // -------------------------------------------------------------------------
    // Patterns
    // -------------------------------------------------------------------------

    parsePat() {
        // sPat | dPat
        // sPat ::= basicExpr
        // dPat ::= destrSpec "{" ... | destrSpec Name ["=" sPat]

        // Heuristic: Look for `{` after a potential prefix.
        // Save pos.
        const save = this.tokPos;
        try {
            // Try parse DestrPat
            const spec = this.parseDestrSpec();

            if (this.peek().txt === '{') {
                // destrSpec "{" {destrField ...} ..
                const node = new FieldDestrPat();
                node.specifiers = spec;
                this.expect('Symbol', '{');
                node.fields = this.parseDelimitedList(() => this.parseFieldDestrField());
                if (this.match('Symbol', '..')) {
                    node.extraFields.present = true;
                    node.extraFields.txt = '..';
                }
                this.expect('Symbol', '}');
                return node;
            }

            if (this.peek().type === 'Name') {
                // destrSpec Name ["=" sPat]
                const node = new VarDestrPat();
                node.base = new UncondVarDestrPat();
                node.base.specifiers = spec;
                node.base.name = this.parseName();

                if (this.match('Symbol', '=')) {
                    node.eq.present = true;
                    node.eq.txt = '=';
                    node.valPat = this.parseSimplePat();
                }
                return node;
            }

        } catch (e) {
            // Fallback to sPat
            this.tokPos = save;
        }

        return this.parseSimplePat();
    }

    parseSimplePat() {
        const node = new SimplePat();
        node.expr = this.parseExpr(); // basicExpr
        return node;
    }

    parseUncondDestrPat() {
        // Similar to pat but specific
        const save = this.tokPos;
        try {
            const spec = this.parseDestrSpec();

            if (this.peek().txt === '{') {
                // Check for UncondFieldDestrField (| | pat) vs UncondPatFieldDestrPat (pat)
                // Grammar:
                // UncondPatFieldDestrPat ::= destrSpec "{" {uncondDestrPat ...} "}"
                // UncondFieldDestrPat ::= destrSpec "{" {destrFieldUncond ...} "}"
                // destrFieldUncond ::= "|" tupleableName "|" uncondDestrPat

                this.expect('Symbol', '{');
                const isUncondField = this.peek().txt === '|';

                if (isUncondField) {
                    const node = new UncondFieldDestrPat();
                    node.specifiers = spec;
                    node.fields = this.parseDelimitedList(() => this.parseUncondFieldDestrField());
                    if (this.match('Symbol', '..')) node.extraFields.present = true;
                    this.expect('Symbol', '}');
                    return node;
                } else {
                    const node = new UncondPatFieldDestrPat();
                    node.specifiers = spec;
                    node.fields = this.parseDelimitedList(() => this.parseUncondDestrPat());
                    if (this.match('Symbol', '..')) node.extraFields.present = true;
                    this.expect('Symbol', '}');
                    return node;
                }
            }

            if (this.peek().type === 'Name') {
                const node = new UncondVarDestrPat();
                node.specifiers = spec;
                node.name = this.parseName();
                return node;
            }
        } catch (e) {
            this.tokPos = save;
        }

        if (this.match('Symbol', '_')) {
            return new AlwaysDestrPat();
        }

        return this.parseSimplePat(); // Fallback
    }

    parseDestrSpec() {
        const node = new DestrSpec();
        // sPat | ["mut"] {typePreop}
        // This is hard to distinguish.
        // Try parsing ops first.
        const save = this.tokPos;
        let ops = [];
        if (this.match('Keyword', 'mut')) {
            // check if followed by typePreOp
            if (this.peek().txt === '*' || this.peek().txt === '&') {
                // consume mut, then ops
                // reset and parse properly
                this.tokPos = save;
                const opNode = new OpDestrSpec();
                opNode.mutKw.txt = this.consume().txt;
                while (this.peek().txt === '*' || this.peek().txt === '&') {
                    opNode.ops.push(this.parseTypePreOp());
                }
                return opNode;
            }
        }

        // Try simple ops
        if (this.peek().txt === '*' || this.peek().txt === '&') {
            const opNode = new OpDestrSpec();
            while (this.peek().txt === '*' || this.peek().txt === '&') {
                opNode.ops.push(this.parseTypePreOp());
            }
            return opNode;
        }

        // SimplePatDestrSpec uses sPat (basicExpr)
        // This is ambiguous with just a Name.
        // We assume if it's not ops, it's a SimplePat (expression).
        const nodeSimple = new SimplePatDestrSpec();
        nodeSimple.type = this.parseSimplePat();
        return nodeSimple;
    }

    parseUncondFieldDestrField() {
        const node = new UncondFieldDestrField();
        this.expect('Symbol', '|');
        if (/[0-9]/.test(this.peek().txt)) {
            node.var = new NumTuplableName();
            node.var.name = this.parseNum();
        } else {
            node.var = new NameTuplableName();
            node.var.name = this.parseName();
        }
        this.expect('Symbol', '|');
        node.pat = this.parseUncondDestrPat();
        return node;
    }

    // -------------------------------------------------------------------------
    // Types & Paths
    // -------------------------------------------------------------------------

    parseTypedParam() {
        const node = new TypedParam();
        if (this.match('Keyword', 'const')) {
            node.constKw.present = true;
            node.constKw.txt = 'const';
        }
        node.name = this.parseName();
        this.expect('Symbol', '=');
        node.type = this.parseExpr(); // Types are exprs
        return node;
    }

    parseModPath() {
        const node = new ModPath();
        const t = this.peek().txt;
        if (t === 'self') {
            node.root = new ModPathRootSelf();
            this.consume();
        } else if (t === 'crate') {
            node.root = new ModPathRootCrate();
            this.consume();
        } else if (t === ':>') {
            node.root = new ModPathRootOp();
            this.consume();
        } else {
            node.root = new ModPathRootName();
            node.root.name = this.parseName();
        }

        while (this.peek().txt === '::') {
            const seg = new ModPathSegment();
            this.consume(); // ::
            seg.name = this.parseName();
            node.segments.push(seg);
        }
        return node;
    }

    parseModPathExpr() {
        const node = new ModPathExpr();
        node.path = this.parseModPath();
        return node;
    }

    parseWhereClauses() {
        if (!this.match('Keyword', 'where')) return new WhereClauses(); // empty
        const node = new WhereClauses();
        node.whereKw.txt = 'where';
        node.clauses = this.parseDelimitedList(() => this.parseWhereClause());
        return node;
    }

    parseWhereClause() {
        const node = new WhereClause();
        if (this.match('Keyword', 'Self')) {
            // Name class handles string
            node.name = new Name();
            node.name.name = 'Self';
        } else {
            node.name = this.parseName();
        }
        this.expect('Symbol', ':');
        node.type = this.parseExpr();
        return node;
    }

    // -------------------------------------------------------------------------
    // Tables & Annotations
    // -------------------------------------------------------------------------

    parseTableConstructor() {
        const node = new TableConstructor();
        this.expect('Symbol', '{');
        node.fields = this.parseDelimitedList(() => this.parseField());
        this.expect('Symbol', '}');
        return node;
    }

    parseField() {
        // Name = expr | expr
        const save = this.tokPos;
        try {
            // Try Name = expr
            if (this.peek().type === 'Name' && this.peek(1).txt === '=') {
                const node = new NamedField();
                node.name = this.parseName();
                this.expect('Symbol', '=');
                node.expr = this.parseExpr();
                return node;
            }
        } catch (e) {
            this.tokPos = save;
        }

        const node = new ExprField();
        node.expr = this.parseExpr();
        return node;
    }

    parseAnnotations(isOuter) {
        const list = [];
        while (true) {
            const start = this.peek().txt;
            if ((isOuter && (start === '--<' || start === '@<')) || (!isOuter && (start === '@' || start === '---'))) {
                list.push(this.parseAnnotation());
            } else {
                break;
            }
        }
        return list;
    }

    parseAnnotation() {
        const t = this.peek();
        if (t.txt === '@') {
            const node = new ModPathAnnotation();
            this.consume();
            node.path = this.parseModPath();
            if (this.peek().txt === '{') {
                node.table.present = true;
                node.table.tbl = this.parseTableConstructor();
            }
            return node;
        }
        if (t.txt === '---') {
            const node = new DocLineAnnotation();
            const tok = this.consume();
            // Content was captured in tokenizer if we implemented that, else it's part of txt?
            // My tokenizer implementation put content in `.content`.
            node.content = tok.content || "";
            return node;
        }
        if (t.txt === '@<') {
            const node = new OuterModPathAnnotation();
            this.consume();
            node.path = this.parseModPath();
            if (this.peek().txt === '{') {
                node.table.present = true;
                node.table.tbl = this.parseTableConstructor();
            }
            return node;
        }
        if (t.txt === '--<') {
            const node = new OuterDocLineAnnotation();
            const tok = this.consume();
            node.content = tok.content || "";
            return node;
        }
        throw new Error("Expected annotation");
    }

    // -------------------------------------------------------------------------
    // Macros
    // -------------------------------------------------------------------------

    checkMacroInvoke() {
        // modpath "!" ...
        // Name is part of modpath. Check if Name followed by !
        if (this.peek().type === 'Name' && this.peek(1).txt === '!') return true;
        if (this.peek().txt === 'crate' && this.peek(1).txt === '!') return true;
        return false;
    }

    parseMacroInvoke() {
        // macroInvoke ::= modpath "!" macroArgs
        // Returns a generic node or MacroInvoke if defined.
        // Using a generic CompoundNode for MacroInvoke since it wasn't in the provided list explicitly.
        const node = new CompoundNode("MacroInvoke");
        node.path = this.parseModPath();
        this.expect('Symbol', '!');

        // macroArgs
        if (this.match('Symbol', '(')) {
            node.args = new ParenArgs();
            // Parse ProgrammableArgs (assumed similar to expr list for now)
            node.args.args = this.parseDelimitedList(() => this.parseExpr());
            this.expect('Symbol', ')');
        } else if (this.match('Symbol', '{')) {
            // Block
            const block = new BlockNode();
            while (!this.match('Symbol', '}')) {
                block.stats.push(this.parseStat());
            }
            node.args = block;
        } else if (this.match('Symbol', '[')) {
            // List
            const list = [];
            while (!this.match('Symbol', ']')) {
                list.push(this.parseExpr());
                if (this.peek().txt === ',') this.consume();
            }
            node.args = list;
        } else {
            // expr | stat | retstat
            // Try expr
            try {
                node.args = this.parseExpr();
            } catch (e) {
                // Try stat
                node.args = this.parseStat();
            }
        }

        return node;
    }

    // -------------------------------------------------------------------------
    // Primitives
    // -------------------------------------------------------------------------

    parseName() {
        // Name or macroInvSpl
        // macroInvSpl ::= modpath "!" macroArgs
        if (this.checkMacroInvoke()) {
            return this.parseMacroInvoke(); // Return macro as Name?
        }

        const node = new Name();
        const tok = this.expect('Name');
        node.name = tok.txt;
        return node;
    }

    parseNum() {
        const node = new Num();
        const tok = this.expect('Numeral');
        node.num = tok.txt;
        return node;
    }

    parseNumFromTok(tok) {
        const node = new Num();
        node.num = tok.txt;
        return node;
    }

    parseSafety() {
        const node = new Safety();
        if (this.match('Keyword', 'safe')) {
            node.kind = 'safe';
        } else if (this.match('Keyword', 'unsafe')) {
            node.kind = 'unsafe';
        } else {
            node.kind = 'default';
        }
        return node;
    }

    // -------------------------------------------------------------------------
    // Generic Utils
    // -------------------------------------------------------------------------

    parseDelimitedList(parser, allowTrailingSpread = false) {
        const list = new DelimitedList();
        while (true) {
            if (allowTrailingSpread && this.peek().txt === '..') {
                // Spread is handled outside usually, but DelimitedListItem is for items.
                // The grammar `... [".."]` suggests spread is part of the container logic, not the list.
                break;
            }
            if (this.peek().txt === '}' || this.peek().txt === ')' || this.peek().txt === ']') break;

            const item = new DelimitedListItem();
            item.value = parser();

            if (this.match('Symbol', ',') || this.match('Symbol', ';')) {
                item.sep.txt = this.tokens[this.tokPos - 1].txt;
                item.sep.present = true;
            } else {
                break;
            }
            list.items.push(item);
        }
        return list;
    }
}
