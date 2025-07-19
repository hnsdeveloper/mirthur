use super::compiler::Instructions;
use super::constants::*;
use super::cpu::*;
use num_traits::int::PrimInt;
use strum::IntoEnumIterator;

trait U8ByteArray {
    fn length(&self) -> usize;
    fn get(&self, n: usize) -> u8;
}

impl<const N: usize> U8ByteArray for [u8; N] {
    fn length(&self) -> usize {
        self.len()
    }

    fn get(&self, n: usize) -> u8 {
        self[n]
    }
}

trait IntegerOps
where
    Self: Sized,
{
    type Bytes: U8ByteArray;
    fn overflowing_add(self, rhs: Self) -> (Self, bool);
    fn overflowing_sub(self, rhs: Self) -> (Self, bool);
    fn overflowing_mul(self, rhs: Self) -> (Self, bool);
    fn as_u64(self) -> u64;
    fn as_le_bytes(self) -> Self::Bytes;
    fn from_le_slice(bytes: &[u8]) -> Result<Self, String>;
}

impl IntegerOps for u8 {
    type Bytes = [u8; size_of::<u8>()];
    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        self.overflowing_add(rhs)
    }

    fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        self.overflowing_sub(rhs)
    }

    fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
        self.overflowing_mul(rhs)
    }

    fn as_le_bytes(self) -> Self::Bytes {
        self.to_le_bytes()
    }

    fn from_le_slice(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() >= size_of::<Self>() {
            return Ok(u8::from_le_bytes(bytes.try_into().unwrap()));
        }
        Err(String::from(
            "Attempting to convert from smaller byte slice",
        ))
    }

    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl IntegerOps for u16 {
    type Bytes = [u8; size_of::<u16>()];
    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        self.overflowing_add(rhs)
    }

    fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        self.overflowing_sub(rhs)
    }

    fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
        self.overflowing_mul(rhs)
    }

    fn as_le_bytes(self) -> Self::Bytes {
        self.to_le_bytes()
    }

    fn from_le_slice(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() >= size_of::<Self>() {
            return Ok(u16::from_le_bytes(bytes.try_into().unwrap()));
        }
        Err(String::from(
            "Attempting to convert from smaller byte slice",
        ))
    }

    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl IntegerOps for u32 {
    type Bytes = [u8; size_of::<u32>()];
    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        self.overflowing_add(rhs)
    }

    fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        self.overflowing_sub(rhs)
    }

    fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
        self.overflowing_mul(rhs)
    }

    fn as_le_bytes(self) -> Self::Bytes {
        self.to_le_bytes()
    }

    fn from_le_slice(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() >= size_of::<Self>() {
            return Ok(u32::from_le_bytes(bytes.try_into().unwrap()));
        }
        Err(String::from(
            "Attempting to convert from smaller byte slice",
        ))
    }

    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl IntegerOps for u64 {
    type Bytes = [u8; size_of::<u64>()];
    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        self.overflowing_add(rhs)
    }

    fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        self.overflowing_sub(rhs)
    }

    fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
        self.overflowing_mul(rhs)
    }

    fn as_le_bytes(self) -> Self::Bytes {
        self.to_le_bytes()
    }

    fn from_le_slice(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() >= size_of::<Self>() {
            return Ok(u64::from_le_bytes(bytes.try_into().unwrap()));
        }
        Err(String::from(
            "Attempting to convert from smaller byte slice",
        ))
    }

    fn as_u64(self) -> u64 {
        self
    }
}

pub struct Program {
    memory: Vec<u8>,
    instructions_size: usize,
    instructions_area_end: usize,
    stack_size: usize,
    registers: Vec<u64>,
}

impl Program {
    pub fn build(mut instructions: Instructions, stack_size: usize) -> Self {
        let registers: Vec<u64> = Reg::iter().map(|_| 0).collect();
        let instructions_size = instructions.len();
        let instructions_end = instructions.len() + instructions.len() % 8;
        instructions.resize(instructions_end + stack_size, 0);
        Self {
            memory: instructions,
            instructions_size,
            instructions_area_end: instructions_end,
            stack_size,
            registers,
        }
    }

    fn get_register(&mut self, r: Reg) -> &mut u64 {
        &mut self.registers[r as u8 as usize]
    }

    fn read_register<T: PrimInt>(&mut self, r: Reg) -> T {
        T::from(*self.get_register(r)).unwrap()
    }

    fn write_register<T: PrimInt + IntegerOps>(&mut self, r: Reg, value: T) {
        *self.get_register(r) = value.as_u64();
    }

    fn increase_pc(&mut self, amount: usize) {
        let mut pc = self.read_register::<u64>(Reg::PC);
        pc += amount as u64;
        self.write_register(Reg::PC, pc);
    }

    fn set_pc(&mut self, addr: u64) {
        let pc = self.get_register(Reg::PC);
        *pc = addr;
    }

    fn add<T: PrimInt + IntegerOps>(&mut self, r1: Reg, r2: Reg) {
        let r1_v = self.read_register::<T>(r1);
        let r2_v = self.read_register::<T>(r2);
        let (r, ov) = r1_v.overflowing_add(r2_v);
        self.write_register(r1, r);
        if ov {
            self.write_register(Reg::FL, OVERFLOW);
        }
    }

    fn sub<T: PrimInt + IntegerOps>(&mut self, r1: Reg, r2: Reg) {
        let r1_v = self.read_register::<T>(r1);
        let r2_v = self.read_register::<T>(r2);
        let (r, ov) = r1_v.overflowing_sub(r2_v);
        self.write_register(r1, r);
        if ov {
            self.write_register(Reg::FL, OVERFLOW);
        }
    }

    fn mul<T: PrimInt + IntegerOps>(&mut self, r1: Reg, r2: Reg) {
        let r1_v = self.read_register::<T>(r1);
        let r2_v = self.read_register::<T>(r2);
        let (r, ov) = r1_v.overflowing_mul(r2_v);
        self.write_register(r1, r);
        if ov {
            self.write_register(Reg::FL, OVERFLOW);
        }
    }

    fn div<T: PrimInt + IntegerOps>(&mut self, r1: Reg, r2: Reg) -> Result<(), String> {
        let r1_v = self.read_register::<T>(r1);
        let r2_v = self.read_register::<T>(r2);
        if r2_v.is_zero() {
            return Err(String::from("Division by zero error."));
        }
        self.write_register(r1, r1_v / r2_v);
        Ok(())
    }

    fn modulo<T: PrimInt + IntegerOps>(&mut self, r1: Reg, r2: Reg) -> Result<(), String> {
        let r1_v = self.read_register::<T>(r1);
        let r2_v = self.read_register::<T>(r2);
        if r2_v.is_zero() {
            return Err(String::from("Division by zero error."));
        }
        self.write_register(r1, r1_v % r2_v);
        Ok(())
    }

    fn pow<T: PrimInt + IntegerOps>(&mut self, r1: Reg, r2: Reg) {
        let r1_v = self.read_register::<T>(r1);
        let r2_v = self.read_register::<T>(r2);
        let mul = r1_v;
        let mut ov = false;
        let mut v: T = T::from(1).unwrap();
        let mut i = T::from(0).unwrap();
        while i != r2_v {
            let (r, over) = v.overflowing_mul(mul);
            v = r;
            ov |= over;
            i = i + T::from(1).unwrap();
        }
        self.write_register(r1, v);
        if ov {
            self.write_register(Reg::FL, OVERFLOW);
        }
    }

    fn write_mem<T: PrimInt + IntegerOps>(&mut self, val: T, addr: u64) -> Result<(), String> {
        if (addr as usize) < self.instructions_area_end
            || addr as usize + size_of::<T>() > self.memory.len()
        {
            return Err(String::from("Attempt to write to invalid memory."));
        }
        let val = val.as_le_bytes();
        for i in 0..val.length() {
            self.memory[addr as usize + i] = val.get(i);
        }
        Ok(())
    }

    fn read_mem<T: PrimInt + IntegerOps>(&mut self, addr: u64) -> Result<T, String> {
        if (addr as usize + size_of::<T>()) >= self.memory.len() {
            return Err("Attempting to read from invalid memory.".into());
        }
        let result = T::from_le_slice(&self.memory[addr as usize..])?;
        Ok(result)
    }

    pub fn run(&mut self) -> Result<u64, String> {
        let ie = self.instructions_area_end;
        let sp = ie + self.stack_size;
        self.write_register(Reg::SP, sp as u64);
        loop {
            let pc = self.read_register::<u64>(Reg::PC);
            if pc as usize >= self.instructions_size {
                return Err("Invalid PC".into());
            }
            let (instruction, size) = Instruction::deserialize(&self.memory[pc as usize..])?;
            match instruction {
                Instruction::Add(r1, r2) => {
                    self.write_register::<u64>(Reg::FL, 0);
                    self.add::<u64>(r1, r2);
                    self.increase_pc(size);
                }
                Instruction::Sub(r1, r2) => {
                    self.write_register::<u64>(Reg::FL, 0);
                    self.sub::<u64>(r1, r2);
                    self.increase_pc(size);
                }
                Instruction::Mul(r1, r2) => {
                    self.write_register::<u64>(Reg::FL, 0);
                    self.mul::<u64>(r1, r2);
                    self.increase_pc(size);
                }
                Instruction::Div(r1, r2) => {
                    self.write_register::<u64>(Reg::FL, 0);
                    self.div::<u64>(r1, r2)?;
                    self.increase_pc(size);
                }
                Instruction::Mod(r1, r2) => {
                    self.write_register::<u64>(Reg::FL, 0);
                    self.modulo::<u64>(r1, r2)?;
                    self.increase_pc(size);
                }
                Instruction::Pow(r1, r2) => {
                    self.write_register::<u64>(Reg::FL, 0);
                    self.pow::<u64>(r1, r2);
                    self.increase_pc(size);
                }
                Instruction::Mov(r1, r2) => {
                    let v = self.read_register::<u64>(r2);
                    self.write_register(r1, v);
                    self.increase_pc(size);
                }
                Instruction::StoreMem(r, addr) => {
                    let d = self.read_register::<u64>(r);
                    self.write_mem(d, addr)?;
                    self.increase_pc(size);
                }
                Instruction::StoreReg(r_what, r_where, offset) => {
                    let addr = self.read_register::<u64>(r_where) + offset;
                    let d: u64 = self.read_register(r_what);
                    self.write_mem(d, addr)?;
                    self.increase_pc(size);
                }
                Instruction::LoadMem(r, addr) => {
                    let mem = self.read_mem::<u64>(addr)?;
                    self.write_register(r, mem);
                    self.increase_pc(size);
                }
                Instruction::LoadReg(r_which, r_where, offset) => {
                    let addr = self.read_register::<u64>(r_where) + offset;
                    let mem = self.read_mem::<u64>(addr)?;
                    self.write_register(r_which, mem);
                    self.increase_pc(size);
                }
                Instruction::MovImm(reg, imm) => {
                    self.write_register(reg, imm);
                    self.increase_pc(size);
                }
                Instruction::Cmp(r1, r2) => {
                    let r1 = self.read_register::<u64>(r1);
                    let r2 = self.read_register::<u64>(r2);
                    let mut flags: u64 = 0;
                    if r1 == r2 {
                        flags |= ZERO_FLAG;
                    }
                    if r1 > r2 {
                        flags |= GT_FLAG;
                    }
                    if r1 < r2 {
                        flags |= LT_FLAG;
                    }
                    self.write_register(Reg::FL, flags);
                    self.increase_pc(size);
                }
                Instruction::Branch(addr) => {
                    self.set_pc(addr);
                }
                Instruction::BranchNe(addr) => {
                    let flags = self.read_register::<u64>(Reg::FL);
                    if flags & ZERO_FLAG == 0 {
                        self.set_pc(addr);
                    } else {
                        self.increase_pc(size);
                    }
                }
                Instruction::BranchEq(addr) => {
                    let flags = self.read_register::<u64>(Reg::FL);
                    if flags & ZERO_FLAG == ZERO_FLAG {
                        self.set_pc(addr);
                    } else {
                        self.increase_pc(size);
                    }
                }
                Instruction::BranchGt(addr) => {
                    let flags = self.read_register::<u64>(Reg::FL);
                    if flags & GT_FLAG == GT_FLAG {
                        self.set_pc(addr);
                    } else {
                        self.increase_pc(size);
                    }
                }
                Instruction::BranchLt(addr) => {
                    let flags = self.read_register::<u64>(Reg::FL);
                    if flags & LT_FLAG == LT_FLAG {
                        self.set_pc(addr);
                    } else {
                        self.increase_pc(size);
                    }
                }
                Instruction::PushReg(r) => {
                    let mut sp = self.read_register::<u64>(Reg::SP);
                    if (sp - size_of::<u64>() as u64) < (self.instructions_area_end as u64) {
                        return Err("Stack overflow".into());
                    }
                    sp -= size_of::<u64>() as u64;
                    let what = self.read_register::<u64>(r);
                    self.write_mem(what, sp)?;
                    self.write_register(Reg::SP, sp);
                    self.increase_pc(size);
                }
                Instruction::PushVal(val) => {
                    let mut sp = self.read_register::<u64>(Reg::SP);
                    if (sp - size_of::<u64>() as u64) < (self.instructions_area_end as u64) {
                        return Err("Stack overflow".into());
                    }
                    sp -= size_of::<u64>() as u64;
                    self.write_mem(val, sp)?;
                    self.write_register(Reg::SP, sp);
                    self.increase_pc(size);
                }
                Instruction::Pop => {
                    let sp = self.read_register::<u64>(Reg::SP) + size_of::<u64>() as u64;
                    if sp > (self.instructions_area_end + self.stack_size) as u64 {
                        return Err("Stack underflow".into());
                    }
                    self.write_register(Reg::SP, sp);
                    self.increase_pc(size);
                }
                Instruction::PopReg(r1) => {
                    let mut sp = self.read_register::<u64>(Reg::SP);
                    if sp + size_of::<u64>() as u64
                        > (self.instructions_area_end + self.stack_size) as u64
                    {
                        return Err("Stack underflow".into());
                    }
                    let mem = self.read_mem::<u64>(sp)?;
                    self.write_register(r1, mem);
                    sp += size_of::<u64>() as u64;
                    self.write_register(Reg::SP, sp);
                    self.increase_pc(size);
                }
                Instruction::Call(addr) => {
                    let new_ra =
                        self.read_register::<u64>(Reg::PC) + instruction.serialize().len() as u64;
                    self.write_register(Reg::RA, new_ra);
                    self.write_register(Reg::PC, addr);
                }
                Instruction::Return => {
                    let ra = self.read_register::<u64>(Reg::RA);
                    self.write_register(Reg::PC, ra);
                }
                Instruction::Halt => {
                    break;
                }
            }
        }
        Ok(*self.get_register(Reg::A0))
    }
}
